"""Tareas: crear un turno (202 + job_id), consultar su estado (polling o stream SSE) y aprobar las críticas.

Ningún endpoint espera al grafo: solo leen o escriben Redis (cliente async) y encolan. Por eso
un POST tarda milisegundos aunque la corrida del grafo tarde minutos.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from app.api.deps import RecursosDep, requiere_revisor
from app.api.schemas import (
    PATRON_JOB_ID,
    AprobacionAceptada,
    ErrorOut,
    Links,
    TareaAceptada,
    TareaIn,
)
from app.worker.jobs import ConversacionOcupada, JobStore
from app.worker.models import FINALES, DecisionHumana, Estado, JobRecord, MensajeReanudar

router = APIRouter(prefix="/tasks")

JobId = Annotated[str, Path(pattern=PATRON_JOB_ID, description="Id del job (32 caracteres hex).")]


def _links(job_id: str, thread_id: str, esperando: bool = False) -> Links:
    return Links(
        estado=f"/tasks/{job_id}",
        conversacion=f"/conversations/{thread_id}",
        aprobar=f"/tasks/{job_id}/approve" if esperando else None,
    )


@router.post(
    "",
    tags=["tareas"],
    summary="Crear un turno (responde 202 con job_id y thread_id)",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TareaAceptada,
    responses={409: {"model": ErrorOut, "description": "La conversación ya tiene un job en curso."}},
)
async def crear_tarea(tarea: TareaIn, recursos: RecursosDep, response: Response) -> TareaAceptada:
    """Encola un turno y devuelve el `job_id` (y el `thread_id` de la conversación) sin esperar al grafo.

    Mandá el mismo `thread_id` para hacer una pregunta de seguimiento sobre la misma conversación.
    """
    try:
        job = await recursos.store.crear(tarea.pregunta, tarea.publicar, tarea.thread_id)
    except ConversacionOcupada as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{exc} Esperá a que termine (o aprobalo) y volvé a enviar.") from exc
    await recursos.worker.encolar_ejecucion(job.job_id)
    response.headers["Location"] = f"/tasks/{job.job_id}"
    return TareaAceptada(job_id=job.job_id, thread_id=job.thread_id, status=job.status, links=_links(job.job_id, job.thread_id))


@router.get(
    "/{job_id}",
    tags=["tareas"],
    summary="Estado, resultado y métricas de un job",
    response_model=JobRecord,
    responses={404: {"model": ErrorOut}},
)
async def estado_tarea(job_id: JobId, recursos: RecursosDep) -> JobRecord:
    """Estado actual del job: PENDING, RUNNING, WAITING_APPROVAL, DONE o FAILED (lectura de Redis)."""
    job = await recursos.store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    return job


INTERVALO_STREAM_S = 0.25  # cada cuánto se miran los eventos nuevos en Redis
LATIDO_S = 15.0  # comentario SSE para que proxies y clientes no corten una conexión sin novedades
CORTES_STREAM = FINALES | {Estado.WAITING_APPROVAL}


def _sse(evento: str, datos: str) -> str:
    return f"event: {evento}\ndata: {datos}\n\n"


async def _eventos_sse(job_id: str, store: JobStore, request: Request) -> AsyncIterator[str]:
    """Transmite tokens de la síntesis, estados y nodos nuevos hasta DONE/FAILED o WAITING_APPROVAL."""
    vistos_estados = vistos_pasos = vistos_borrador = 0
    sin_novedades_s = 0.0
    while True:
        job = await store.get(job_id)
        if job is None:  # venció el TTL o se borró mientras se transmitía
            return
        # Primero el texto nuevo de la síntesis: el nodo `synthesizer` "termina" después de escribirlo.
        borrador = await store.leer_borrador(job_id)
        if len(borrador) < vistos_borrador:  # el modelo reintentó y el borrador volvió a empezar
            yield _sse("reinicio", "{}")
            vistos_borrador = 0
        texto_nuevo = borrador[vistos_borrador:]
        if texto_nuevo:
            yield _sse("token", json.dumps({"texto": texto_nuevo}, ensure_ascii=False))
            vistos_borrador = len(borrador)
        nuevos = [("estado", e.t, e.model_dump_json(exclude_none=True)) for e in job.eventos[vistos_estados:]]
        nuevos += [("nodo", p.t, p.model_dump_json(exclude_none=True)) for p in job.pasos[vistos_pasos:]]
        vistos_estados, vistos_pasos = len(job.eventos), len(job.pasos)
        for evento, _, datos in sorted(nuevos, key=lambda n: n[1]):
            yield _sse(evento, datos)
        if job.status in CORTES_STREAM:
            # El último evento trae el job completo (resultado o lo que hay que aprobar).
            yield _sse("fin", job.model_dump_json(exclude={"eventos", "pasos"}))
            return
        sin_novedades_s = 0.0 if nuevos or texto_nuevo else sin_novedades_s + INTERVALO_STREAM_S
        if sin_novedades_s >= LATIDO_S:
            sin_novedades_s = 0.0
            yield ": latido\n\n"
        if await request.is_disconnected():
            return
        await asyncio.sleep(INTERVALO_STREAM_S)


@router.get(
    "/{job_id}/stream",
    tags=["tareas"],
    summary="Progreso en vivo de un job (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Eventos `estado` (EventoEstado), `nodo` (EventoNodo) y un `fin` final con el JobRecord.",
        },
        404: {"model": ErrorOut},
    },
)
async def stream_tarea(job_id: JobId, recursos: RecursosDep, request: Request) -> StreamingResponse:
    """Alternativa al polling: una conexión que recibe cada cambio de estado y cada nodo del grafo que termina
    (supervisor, investigador, analista, síntesis...). Se cierra sola al terminar o al pausarse para aprobación.

    Probalo con `curl -N localhost:8000/tasks/{job_id}/stream`.
    """
    if await recursos.store.get(job_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    return StreamingResponse(
        _eventos_sse(job_id, recursos.store, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("", tags=["tareas"], summary="Últimos jobs", response_model=list[JobRecord])
async def listar_tareas(recursos: RecursosDep, limite: int = Query(20, ge=1, le=100)) -> list[JobRecord]:
    """Los últimos jobs, del más nuevo al más viejo."""
    return await recursos.store.listar(limite)


@router.post(
    "/{job_id}/approve",
    summary="Aprobar o rechazar una tarea crítica pausada",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AprobacionAceptada,
    tags=["human-in-the-loop"],
    dependencies=[Depends(requiere_revisor)],
    responses={
        401: {"model": ErrorOut, "description": "Falta X-Reviewer-Key o no es válida (si hay REVIEWER_API_KEY)."},
        404: {"model": ErrorOut, "description": "No existe el job."},
        409: {"model": ErrorOut, "description": "El job no está en WAITING_APPROVAL (o ya se aprobó)."},
    },
)
async def aprobar(job_id: JobId, decision: DecisionHumana, recursos: RecursosDep) -> AprobacionAceptada:
    """Registra la decisión humana y encola la reanudación desde el checkpoint (no espera a que termine)."""
    job = await recursos.store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}.")
    # Transición atómica WAITING_APPROVAL -> PENDING: dos aprobaciones simultáneas no pueden
    # reanudar dos veces el mismo hilo (la segunda recibe 409).
    if not await recursos.store.tomar_aprobacion(job_id, decision):
        actual = await recursos.store.get(job_id)
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job {job_id} no está esperando aprobación (estado: {actual.status if actual else '?'}).",
        )
    await recursos.worker.encolar_reanudacion(MensajeReanudar(job_id=job_id, decision=decision))
    return AprobacionAceptada(
        job_id=job_id,
        thread_id=job.thread_id,
        status=Estado.PENDING,
        aprobado=decision.aprobado,
        links=_links(job_id, job.thread_id),
    )
