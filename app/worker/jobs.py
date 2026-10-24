"""Estado de los jobs y de las conversaciones en Redis (cliente `redis.asyncio`).

Claves (con el prefijo `REDIS_PREFIJO`, por defecto `final`):

    final:job:{job_id}              hash con el estado del job (lo valida `JobRecord`)
    final:job:{job_id}:eventos      lista con cada transición de estado y su hora
    final:job:{job_id}:pasos        lista con cada nodo del grafo que terminó (streaming del progreso)
    final:job:{job_id}:borrador     texto que va escribiendo el modelo de síntesis (streaming de tokens)
    final:jobs                      sorted set por fecha de creación (GET /tasks)
    final:thread:{thread_id}:jobs   lista con los jobs (turnos) de una conversación
    final:thread:{thread_id}:activo el job que está usando la conversación ahora (candado)

El estado del job (para el cliente que hace polling) y el CHECKPOINT del grafo (para LangGraph)
son cosas distintas y viven en claves distintas del mismo Redis: el job dice "en qué anda este
turno"; el checkpoint (`checkpoint:{thread_id}:*`) guarda el estado completo de la conversación.

Candado por conversación: dos turnos del mismo `thread_id` no pueden correr a la vez (los dos
escribirían el mismo checkpoint y el segundo no vería la respuesta del primero). Si hay un job
activo en el hilo, `crear` levanta `ConversacionOcupada` y la API responde 409.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.worker.models import FINALES, DecisionHumana, Estado, EventoNodo, JobRecord

_CAMPOS_JSON = ("resultado", "aprobacion_pendiente", "aprobacion", "metricas")


class ConversacionOcupada(Exception):
    def __init__(self, thread_id: str, job_activo: str) -> None:
        super().__init__(f"La conversación {thread_id} tiene un job en curso ({job_activo}).")
        self.thread_id = thread_id
        self.job_activo = job_activo


def ahora() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def segundos_entre(desde: str, hasta: str) -> float:
    return round((datetime.fromisoformat(hasta) - datetime.fromisoformat(desde)).total_seconds(), 3)


class JobStore:
    """Lectura y escritura del estado de los jobs. Todas las operaciones son async."""

    def __init__(self, redis: Redis, prefijo: str = "final", ttl: int = 7 * 24 * 3600) -> None:
        # El cliente tiene que crearse con decode_responses=True (str, no bytes).
        self.redis = redis
        self.prefijo = prefijo
        self.ttl = ttl

    # --- claves ---

    def clave(self, job_id: str) -> str:
        return f"{self.prefijo}:job:{job_id}"

    def _eventos(self, job_id: str) -> str:
        return f"{self.clave(job_id)}:eventos"

    def _pasos(self, job_id: str) -> str:
        return f"{self.clave(job_id)}:pasos"

    def _borrador(self, job_id: str) -> str:
        return f"{self.clave(job_id)}:borrador"

    def _jobs_hilo(self, thread_id: str) -> str:
        return f"{self.prefijo}:thread:{thread_id}:jobs"

    def _candado(self, thread_id: str) -> str:
        return f"{self.prefijo}:thread:{thread_id}:activo"

    # --- jobs ---

    async def crear(self, pregunta: str, publicar: bool = False, thread_id: str | None = None) -> JobRecord:
        """Crea el job en PENDING. Sin `thread_id`, abre una conversación nueva."""
        job_id = uuid.uuid4().hex
        thread_id = thread_id or uuid.uuid4().hex
        await self._tomar_candado(thread_id, job_id)
        momento = ahora()
        campos = {
            "job_id": job_id,
            "thread_id": thread_id,
            "status": Estado.PENDING,
            "pregunta": pregunta,
            "publicar": int(publicar),
            "created_at": momento,
            "updated_at": momento,
        }
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self.clave(job_id), mapping=campos)
            pipe.expire(self.clave(job_id), self.ttl)
            pipe.rpush(self._eventos(job_id), json.dumps({"status": Estado.PENDING, "t": momento}))
            pipe.expire(self._eventos(job_id), self.ttl)
            pipe.zadd(f"{self.prefijo}:jobs", {job_id: datetime.now(UTC).timestamp()})
            pipe.rpush(self._jobs_hilo(thread_id), job_id)
            await pipe.execute()
        return await self.get(job_id)  # type: ignore[return-value]

    async def get(self, job_id: str) -> JobRecord | None:
        datos: dict[str, Any] = await self.redis.hgetall(self.clave(job_id))
        if not datos:
            return None
        for campo in _CAMPOS_JSON:
            if campo in datos:
                datos[campo] = json.loads(datos[campo])
        datos["publicar"] = datos.get("publicar") == "1"
        datos["eventos"] = [json.loads(e) for e in await self.redis.lrange(self._eventos(job_id), 0, -1)]
        datos["pasos"] = [json.loads(p) for p in await self.redis.lrange(self._pasos(job_id), 0, -1)]
        return JobRecord.model_validate(datos)

    async def actualizar(self, job_id: str, status: Estado | None = None, **campos: Any) -> None:
        """Cambia el estado y/o campos. Modelos Pydantic y dict/list se guardan como JSON; None borra el campo."""
        momento = ahora()
        valores: dict[str, Any] = {"updated_at": momento}
        borrar = [k for k, v in campos.items() if v is None]
        for k, v in campos.items():
            if isinstance(v, BaseModel):
                valores[k] = v.model_dump_json()
            elif isinstance(v, (dict, list)):
                valores[k] = json.dumps(v, ensure_ascii=False)
            elif v is not None:
                valores[k] = v
        if status is not None:
            valores["status"] = status
            if status in FINALES:
                valores["finished_at"] = momento
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self.clave(job_id), mapping=valores)
            if borrar:
                pipe.hdel(self.clave(job_id), *borrar)
            if status is not None:
                pipe.rpush(self._eventos(job_id), json.dumps({"status": status, "t": momento}))
            await pipe.execute()
        if status == Estado.RUNNING:
            # started_at es la PRIMERA vez que corrió: no se pisa al reanudar.
            await self.redis.hsetnx(self.clave(job_id), "started_at", momento)

    async def registrar_paso(self, job_id: str, nodo: str, detalle: str | None = None) -> None:
        """Agrega un nodo terminado al progreso del job (lo lee el stream SSE)."""
        paso = EventoNodo(nodo=nodo, t=ahora(), detalle=detalle)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.rpush(self._pasos(job_id), paso.model_dump_json(exclude_none=True))
            pipe.expire(self._pasos(job_id), self.ttl)
            await pipe.execute()

    async def agregar_borrador(self, job_id: str, texto: str, reiniciar: bool = False) -> None:
        """Suma tokens al borrador de la respuesta. `reiniciar` lo vacía antes (reintento del modelo)."""
        async with self.redis.pipeline(transaction=True) as pipe:
            if reiniciar:
                pipe.delete(self._borrador(job_id))
            pipe.append(self._borrador(job_id), texto)
            pipe.expire(self._borrador(job_id), self.ttl)
            await pipe.execute()

    async def leer_borrador(self, job_id: str) -> str:
        return await self.redis.get(self._borrador(job_id)) or ""

    async def tomar_aprobacion(self, job_id: str, decision: DecisionHumana) -> bool:
        """WAITING_APPROVAL -> PENDING de forma atómica (WATCH/MULTI). False si no estaba esperando."""
        clave = self.clave(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(clave)
                if await pipe.hget(clave, "status") != Estado.WAITING_APPROVAL:
                    return False
                momento = ahora()
                pipe.multi()
                pipe.hset(
                    clave, mapping={"status": Estado.PENDING, "updated_at": momento, "aprobacion": decision.model_dump_json()}
                )
                pipe.rpush(self._eventos(job_id), json.dumps({"status": Estado.PENDING, "t": momento, "por": "approve"}))
                await pipe.execute()
                return True
            except WatchError:  # otro cliente cambió el job entre el WATCH y el EXEC
                return False

    async def listar(self, limite: int = 20) -> list[JobRecord]:
        ids = await self.redis.zrevrange(f"{self.prefijo}:jobs", 0, limite - 1)
        jobs = [await self.get(job_id) for job_id in ids]
        return [j for j in jobs if j is not None]

    # --- conversaciones ---

    async def jobs_del_hilo(self, thread_id: str) -> list[str]:
        return await self.redis.lrange(self._jobs_hilo(thread_id), 0, -1)

    async def job_activo(self, thread_id: str) -> str | None:
        return await self.redis.get(self._candado(thread_id))

    async def _tomar_candado(self, thread_id: str, job_id: str) -> None:
        clave = self._candado(thread_id)
        if await self.redis.set(clave, job_id, nx=True, ex=self.ttl):
            return
        # Ya hay un job en el hilo. Si terminó (o venció) el candado quedó huérfano: se reemplaza
        # con WATCH para que dos pedidos simultáneos no lo tomen los dos.
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(clave)
                activo = await pipe.get(clave)
                if activo:
                    job = await self.get(activo)
                    if job is not None and job.status not in FINALES:
                        raise ConversacionOcupada(thread_id, activo)
                pipe.multi()
                pipe.set(clave, job_id, ex=self.ttl)
                await pipe.execute()
            except WatchError as exc:
                raise ConversacionOcupada(thread_id, "otro pedido simultáneo") from exc

    async def liberar_hilo(self, thread_id: str, job_id: str) -> None:
        """Suelta el candado solo si lo tiene ESTE job (WATCH/MULTI: no borra el de otro)."""
        clave = self._candado(thread_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(clave)
                if await pipe.get(clave) != job_id:
                    return
                pipe.multi()
                pipe.delete(clave)
                await pipe.execute()
            except WatchError:
                return
