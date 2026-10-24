"""Worker: corre el grafo en segundo plano y mantiene el estado del job en Redis.

    POST /tasks ──LPUSH──> final:cola ──BLMOVE──> final:procesando ──> Worker.procesar()
                                                                        │ RUNNING
                                                                        ├─ terminó ............ DONE (+ resultado y métricas)
                                                                        ├─ interrupt() ........ WAITING_APPROVAL (+ qué aprobar)
                                                                        └─ excepción .......... FAILED (+ error)

- La cola vive en Redis, no en memoria: los mensajes sobreviven a un reinicio de la API y se
  pueden levantar workers en otro proceso (`uv run python -m app.worker`).
- `BLMOVE` pasa cada mensaje de la cola a `procesando` en un solo paso atómico; se borra de ahí
  al terminar. Si el proceso muere a mitad de camino, al arrancar `recuperar()` lo devuelve a la
  cola (entrega "al menos una vez") y, como hay checkpoint, el grafo CONTINÚA desde el último
  paso guardado en vez de empezar de cero.
- Cada mensaje se valida con Pydantic (`MENSAJE_COLA`) al leerlo: uno mal formado se descarta
  con un log, no tira abajo al consumidor.
- `concurrencia` consumidores (tareas asyncio) comparten el event loop: el grafo es async de
  punta a punta (LLM, Pinecone, Redis), así que 5 jobs avanzan a la vez sin bloquear la API.
- NINGUNA excepción se escapa sin quedar registrada: el job pasa a FAILED con el tipo y el
  mensaje del error y el candado de la conversación se libera.
"""

import asyncio
import logging
import time
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import ValidationError
from redis.asyncio import Redis

from app.core.config import PrecioModelo
from app.graph.builder import run_config
from app.graph.state import entrada_turno
from app.observability.metricas import Metricas, MetricasLLM
from app.rag.schemas import Fuente
from app.worker.jobs import JobStore, ahora, segundos_entre
from app.worker.models import (
    MENSAJE_COLA,
    AprobacionPendiente,
    ContribucionOut,
    Estado,
    JobRecord,
    MensajeEjecutar,
    MensajeReanudar,
    ResultadoTarea,
)

logger = logging.getLogger(__name__)

MAX_ERROR = 1000  # largo máximo del mensaje de error guardado en el job


def fuentes_del_turno(contribuciones: list[dict[str, Any]]) -> list[Fuente]:
    """Las fuentes del RAG de todas las búsquedas del turno, sin repetir chunks, por score."""
    por_chunk: dict[str, Fuente] = {}
    for c in contribuciones:
        for crudo in c.get("fuentes_rag", []):
            fuente = Fuente.model_validate(crudo)
            previa = por_chunk.get(fuente.chunk_id)
            if previa is None or fuente.score > previa.score:
                por_chunk[fuente.chunk_id] = fuente
    return sorted(por_chunk.values(), key=lambda f: f.score, reverse=True)


def resumen_resultado(valores: dict[str, Any]) -> ResultadoTarea:
    """Lo que ve el cliente en `resultado`: la respuesta y su trazabilidad, sin mensajes internos."""
    contribuciones = valores.get("contributions", [])
    return ResultadoTarea(
        respuesta=valores.get("final_answer", ""),
        fuentes=fuentes_del_turno(contribuciones),
        contribuciones=[ContribucionOut.model_validate(c) for c in contribuciones],
        ruta_supervisor=[d["next"] for d in valores.get("decisiones", [])],
        turno=len(valores.get("historial", [])),
        task_completed=valores.get("task_completed", False),
        cierre_forzado=valores.get("cierre_forzado", False),
        fuera_de_dominio=valores.get("fuera_de_dominio", False),
        motivos_criticidad=valores.get("motivos_criticidad", []),
        aprobacion=valores.get("aprobacion"),
        reporte_publicado=valores.get("reporte_publicado"),
    )


def espera_de_aprobacion(job: JobRecord) -> float | None:
    """Segundos entre WAITING_APPROVAL y la aprobación (sumados, si hubo más de una pausa)."""
    total, desde = 0.0, None
    for evento in job.eventos:
        if evento.status == Estado.WAITING_APPROVAL:
            desde = evento.t
        elif desde and evento.por == "approve":
            total += segundos_entre(desde, evento.t)
            desde = None
    return round(total, 3) if any(e.status == Estado.WAITING_APPROVAL for e in job.eventos) else None


class Worker:
    def __init__(
        self,
        redis: Redis,
        store: JobStore,
        graph: CompiledStateGraph,
        precios: dict[str, PrecioModelo] | None = None,
        concurrencia: int = 5,
        espera_cola_s: float = 1.0,
    ) -> None:
        self.redis = redis
        self.store = store
        self.graph = graph
        self.precios = precios or {}
        self.concurrencia = concurrencia
        self.espera_cola_s = espera_cola_s
        self.cola = f"{store.prefijo}:cola"
        self.procesando = f"{store.prefijo}:procesando"
        self._tareas: list[asyncio.Task] = []
        self._activo = False

    # --- Productor (lo usan los endpoints) ---

    async def encolar_ejecucion(self, job_id: str) -> None:
        await self.redis.lpush(self.cola, MensajeEjecutar(job_id=job_id).model_dump_json())

    async def encolar_reanudacion(self, mensaje: MensajeReanudar) -> None:
        await self.redis.lpush(self.cola, mensaje.model_dump_json())

    # --- Ciclo de vida ---

    async def iniciar(self) -> None:
        await self.recuperar()
        self._activo = True
        self._tareas = [asyncio.create_task(self._consumir(i), name=f"worker-{i}") for i in range(self.concurrencia)]
        logger.info("Worker iniciado con %d consumidores", self.concurrencia)

    async def detener(self) -> None:
        self._activo = False
        for tarea in self._tareas:
            tarea.cancel()
        await asyncio.gather(*self._tareas, return_exceptions=True)
        self._tareas = []

    async def recuperar(self) -> int:
        """Devuelve a la cola los mensajes que quedaron a medio procesar (caída del proceso)."""
        recuperados = 0
        while await self.redis.lmove(self.procesando, self.cola, "RIGHT", "RIGHT"):
            recuperados += 1
        if recuperados:
            logger.warning("Se reencolaron %d mensajes que habían quedado en proceso", recuperados)
        return recuperados

    async def _consumir(self, numero: int) -> None:
        while self._activo:
            try:
                crudo = await self.redis.blmove(self.cola, self.procesando, self.espera_cola_s, "RIGHT", "LEFT")
            except asyncio.CancelledError:
                raise
            except Exception:  # Redis caído: se espera y se reintenta, el consumidor no muere
                logger.exception("worker-%d: no pude leer la cola", numero)
                await asyncio.sleep(1)
                continue
            if crudo is None:
                await asyncio.sleep(0.05)
                continue
            try:
                mensaje = MENSAJE_COLA.validate_json(crudo)
            except ValidationError:
                logger.error("worker-%d: mensaje inválido en la cola, se descarta: %.200s", numero, crudo)
            else:
                # procesar() nunca levanta salvo CancelledError (apagado): en ese caso el mensaje
                # queda en `procesando` a propósito, para que recuperar() lo reencole.
                await self.procesar(mensaje)
            await self.redis.lrem(self.procesando, 1, crudo)

    # --- Ejecución de un job ---

    async def procesar(self, mensaje: MensajeEjecutar | MensajeReanudar) -> None:
        """Corre (o reanuda) el turno de un job. Nunca deja escapar una excepción."""
        job_id = mensaje.job_id
        job: JobRecord | None = None
        metricas_corrida: Metricas | None = None
        try:
            job = await self.store.get(job_id)
            if job is None:
                logger.warning("Mensaje para un job inexistente (%s): se descarta", job_id)
                return
            await self.store.actualizar(job_id, Estado.RUNNING, aprobacion_pendiente=None)
            contador = MetricasLLM(self.precios)
            config = run_config(job.thread_id, job_id, fase=mensaje.tipo, callbacks=[contador])

            if isinstance(mensaje, MensajeReanudar):
                # La decisión humana entra al grafo como valor de retorno del interrupt().
                entrada: Any = Command(resume=mensaje.decision.model_dump())
            else:
                previo = await self.graph.aget_state(config)
                if previo.next and previo.values.get("job_id") == job_id:
                    # Este mismo job ya tenía un checkpoint a mitad de camino (el proceso se cayó): se continúa.
                    entrada = None
                else:
                    # Turno nuevo de la conversación (el primero o un seguimiento).
                    entrada = entrada_turno(job.pregunta, job_id=job_id, publicar=job.publicar)

            inicio = time.perf_counter()
            try:
                # `astream` en vez de `ainvoke`: mismo resultado, pero cada nodo que termina queda
                # registrado en Redis y `GET /tasks/{id}/stream` lo transmite en vivo.
                async for actualizacion in self.graph.astream(entrada, config, stream_mode="updates"):
                    for nodo, cambios in actualizacion.items():
                        if not nodo.startswith("__"):  # "__interrupt__" no es un nodo
                            await self.store.registrar_paso(job_id, nodo, detalle_paso(nodo, cambios))
            finally:
                metricas_corrida = contador.metricas(time.perf_counter() - inicio)

            snapshot = await self.graph.aget_state(config)
            interrupciones = [i for tarea in snapshot.tasks for i in tarea.interrupts]
            metricas = (job.metricas or Metricas()).sumar(metricas_corrida)
            if interrupciones:
                await self.store.actualizar(
                    job_id,
                    Estado.WAITING_APPROVAL,
                    aprobacion_pendiente=AprobacionPendiente.model_validate(interrupciones[0].value),
                    metricas=metricas,
                )
                return
            await self._terminar(job, Estado.DONE, metricas, resultado=resumen_resultado(snapshot.values))
        except asyncio.CancelledError:
            # Apagado ordenado: el job queda como está y su mensaje sigue en `procesando`
            # para que recuperar() lo reencole al volver a arrancar.
            raise
        except Exception as exc:
            logger.exception("El job %s falló", job_id)
            if job is None:
                return
            try:
                metricas = (job.metricas or Metricas()).sumar(metricas_corrida or Metricas())
                await self._terminar(job, Estado.FAILED, metricas, error=f"{type(exc).__name__}: {exc}"[:MAX_ERROR])
            except Exception:  # ni siquiera Redis responde: se registra en el log
                logger.exception("No pude marcar FAILED el job %s", job_id)

    async def _terminar(self, job: JobRecord, estado: Estado, metricas: Metricas, **campos: Any) -> None:
        """DONE o FAILED: completa las latencias, guarda y libera la conversación."""
        actual = await self.store.get(job.job_id) or job
        metricas = metricas.model_copy(
            update={
                "latencia_total_s": segundos_entre(job.created_at, ahora()),
                "espera_aprobacion_s": espera_de_aprobacion(actual),
            }
        )
        await self.store.actualizar(job.job_id, estado, metricas=metricas, **campos)
        await self.store.liberar_hilo(job.thread_id, job.job_id)


async def _main() -> None:
    """Worker como proceso aparte (escala horizontal): `uv run python -m app.worker.worker`.

    Útil con `WORKER_EMBEBIDO=0` en la API: la API solo encola y estos procesos ejecutan.
    """
    from app.core.config import get_settings
    from app.core.logging import configurar_logging
    from app.main import recursos_desde_settings

    settings = get_settings()
    configurar_logging(settings.log_level)
    async with recursos_desde_settings(settings, iniciar_worker=True) as recursos:
        logger.info("Worker standalone con modelos %s; Ctrl+C para salir", recursos.modelos)
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


def detalle_paso(nodo: str, cambios: Any) -> str | None:
    """Resumen legible de lo que hizo un nodo, para el stream de progreso."""
    if not isinstance(cambios, dict):
        return None
    if nodo == "supervisor" and cambios.get("next_agent"):
        decisiones = cambios.get("decisiones") or [{}]
        razon = str(decisiones[-1].get("razon", ""))[:160]
        return f"-> {cambios['next_agent']}" + (f": {razon}" if razon else "")
    contribuciones = cambios.get("contributions") or []
    if contribuciones:
        c = contribuciones[-1]
        fuentes = len(c.get("fuentes") or [])
        return f"{c.get('estado', '?')}" + (f", {fuentes} fuentes" if fuentes else "")
    if cambios.get("motivos_criticidad"):
        return "tarea crítica: pide aprobación humana"
    return None
