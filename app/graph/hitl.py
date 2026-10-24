"""Human-in-the-loop: qué tarea es crítica, el nodo que pausa y el que ejecuta la acción.

CRITERIO DE TAREA CRÍTICA (explícito y testeable, ver `motivos_criticidad`): una tarea es
crítica cuando implica una acción con EFECTOS SECUNDARIOS fuera del sistema: publicar el
reporte final (`publicar_reporte` lo deja en Redis, donde lo leen otros sistemas). Se detecta:
1. si el cliente lo pide explícitamente: `"publicar": true` en el body de `POST /tasks`;
2. si la pregunta lo pide en lenguaje natural ("publicá", "enviá", "notificá", "exportá"...).

Flujo en el grafo:

    synthesizer -> evaluar_criticidad --(no crítica)--> cerrar_turno -> END
                                     --(crítica)----> aprobacion_humana --interrupt()-- PAUSA
    POST /tasks/{id}/approve  -> Command(resume={aprobado, comentario, revisor})   (revisor = dueño de la clave)
                                     aprobacion_humana --(aprobado)--> publicar_reporte -> cerrar_turno
                                                       --(rechazado)-> cerrar_turno (la respuesta lo explica)

`interrupt()` guarda el estado en el checkpoint (Redis) y corta la ejecución. Al reanudar,
LangGraph vuelve a ejecutar `aprobacion_humana` DESDE EL PRINCIPIO y esta vez `interrupt()`
devuelve el valor de `Command(resume=...)`. Por eso el nodo no tiene efectos antes del
`interrupt()`: el efecto secundario vive en otro nodo, al que solo se llega aprobado. El
destino se decide devolviendo `Command(goto=..., update=...)` (ruteo + actualización en uno).
"""

import json
import re
import unicodedata
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt
from redis.asyncio import Redis

from app.graph.state import OrchestratorState
from app.worker.models import DecisionHumana

ACCION_CRITICA = "publicar_reporte"
MAX_BORRADOR = 3000  # lo que ve el revisor del borrador (el resto queda en el checkpoint)
COMO_RESPONDER = 'POST /tasks/{job_id}/approve con el header X-Reviewer-Key y {"aprobado": true|false, "comentario": "..."}'

# Verbos que piden una acción hacia afuera. Se compara sin tildes y en minúsculas:
# "publicá" -> "publica". "publicación" NO matchea (no es un pedido de acción).
_PEDIDO_DE_ACCION = re.compile(r"\b(publica|envia|notifica|exporta)(r|lo|la|los|las)?\b")

# Firma del publicador inyectado: (job_id, reporte) -> destino donde quedó publicado.
Publicador = Callable[[str, dict[str, Any]], Awaitable[str]]


def _sin_tildes(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto.lower())
    return "".join(ch for ch in normalizado if not unicodedata.combining(ch))


def motivos_criticidad(pregunta: str, publicar: bool) -> list[str]:
    """Por qué la tarea es crítica. Lista vacía = no es crítica (no se pausa)."""
    motivos = []
    if publicar:
        motivos.append(f"El cliente pidió '{ACCION_CRITICA}' (publicar=true): acción con efectos secundarios.")
    pedido = _PEDIDO_DE_ACCION.search(_sin_tildes(pregunta))
    if pedido:
        motivos.append(f"La pregunta pide una acción hacia afuera ('{pedido.group(0)}'): se ejecutaría '{ACCION_CRITICA}'.")
    return motivos


async def evaluar_criticidad(state: OrchestratorState) -> dict[str, Any]:
    """Aplica el criterio y deja los motivos en el estado (quedan en la traza y en el checkpoint)."""
    if state.get("fuera_de_dominio"):
        return {"motivos_criticidad": []}  # no hay reporte que publicar
    return {"motivos_criticidad": motivos_criticidad(state["pregunta"], state.get("publicar", False))}


def route_criticidad(state: OrchestratorState) -> Literal["aprobacion_humana", "cerrar_turno"]:
    return "aprobacion_humana" if state.get("motivos_criticidad") else "cerrar_turno"


async def aprobacion_humana(state: OrchestratorState) -> Command[Literal["publicar_reporte", "cerrar_turno"]]:
    """PAUSA obligatoria: nada se publica sin una aprobación externa.

    El payload del `interrupt` es lo que el revisor ve en `GET /tasks/{id}` (`aprobacion_pendiente`).
    """
    decision = interrupt(
        {
            "accion": ACCION_CRITICA,
            "motivos": state.get("motivos_criticidad", []),
            "borrador": state.get("final_answer", "")[:MAX_BORRADOR],
            "como_responder": COMO_RESPONDER,
        }
    )
    # El valor de reanudación entra al grafo desde afuera: se valida con el mismo modelo que la API.
    validada = DecisionHumana.model_validate(decision)
    registro = {**validada.model_dump(), "fecha": datetime.now(UTC).isoformat(timespec="seconds")}
    if registro["aprobado"]:
        return Command(goto=ACCION_CRITICA, update={"aprobacion": registro})

    explicacion = (
        f"La acción '{ACCION_CRITICA}' fue RECHAZADA por revisión humana"
        + (f" (comentario: {registro['comentario']})" if registro["comentario"] else "")
        + ". El reporte NO se publicó; la respuesta de abajo queda solo como borrador."
    )
    return Command(
        goto="cerrar_turno",
        update={
            "aprobacion": registro,
            "final_answer": f"{explicacion}\n\n{state.get('final_answer', '')}",
            "messages": [AIMessage(content=explicacion, name="aprobacion_humana")],
        },
    )


def make_publicar_node(publicador: Publicador):
    """Nodo con el efecto secundario. Solo se llega acá con aprobación humana."""

    async def publicar_reporte(state: OrchestratorState) -> dict[str, Any]:
        job_id = state["job_id"]
        reporte = {
            "job_id": job_id,
            "pregunta": state["pregunta"],
            "respuesta": state["final_answer"],
            "fuentes": sorted({f for c in state.get("contributions", []) if c["agente"] == "researcher" for f in c["fuentes"]}),
            "aprobacion": state.get("aprobacion"),
        }
        destino = await publicador(job_id, reporte)
        return {
            "reporte_publicado": destino,
            "messages": [AIMessage(content=f"Reporte publicado en {destino}.", name=ACCION_CRITICA)],
        }

    return publicar_reporte


def publicador_redis(redis: Redis, prefijo: str) -> Publicador:
    """El efecto secundario real: deja el reporte en `{prefijo}:reporte:{job_id}` y lo anuncia en la
    lista `{prefijo}:reportes_publicados`, de donde lo tomarían otros sistemas (un mail, un Slack...)."""

    async def publicar(job_id: str, reporte: dict[str, Any]) -> str:
        clave = f"{prefijo}:reporte:{job_id}"
        await redis.set(clave, json.dumps(reporte, ensure_ascii=False))
        await redis.lpush(f"{prefijo}:reportes_publicados", job_id)
        return clave

    return publicar
