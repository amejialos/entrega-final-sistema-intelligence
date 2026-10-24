"""Conversaciones: el historial de un `thread_id`, leído del checkpoint de LangGraph en Redis.

Es la prueba visible de la persistencia: después de `docker compose restart api redis`, este
endpoint sigue devolviendo los turnos, porque salen de `graph.aget_state()` (el último
checkpoint del hilo) y no de la memoria del proceso.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, status

from app.api.deps import RecursosDep
from app.api.schemas import PATRON_THREAD_ID, ConversacionOut, ErrorOut, TurnoOut

router = APIRouter(prefix="/conversations", tags=["conversaciones"])

MAX_CHECKPOINTS_CONTADOS = 500


@router.get(
    "/{thread_id}", summary="Historial de una conversación", response_model=ConversacionOut, responses={404: {"model": ErrorOut}}
)
async def ver_conversacion(thread_id: Annotated[str, Path(pattern=PATRON_THREAD_ID)], recursos: RecursosDep) -> ConversacionOut:
    """Turnos respondidos, jobs del hilo, si está pausado y cuántos checkpoints tiene en Redis."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await recursos.graph.aget_state(config)
    jobs = await recursos.store.jobs_del_hilo(thread_id)
    if not snapshot.values and not jobs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe la conversación {thread_id}.")
    checkpoints = 0
    async for _ in recursos.graph.aget_state_history(config, limit=MAX_CHECKPOINTS_CONTADOS):
        checkpoints += 1
    valores = snapshot.values or {}
    return ConversacionOut(
        thread_id=thread_id,
        turnos=[TurnoOut.model_validate(t) for t in valores.get("historial", [])],
        jobs=jobs,
        job_activo=await recursos.store.job_activo(thread_id),
        pausado_en=list(snapshot.next),
        mensajes=len(valores.get("messages", [])),
        checkpoints=checkpoints,
    )
