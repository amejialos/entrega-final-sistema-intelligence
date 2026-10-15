"""Tareas: crear un turno (202 + job_id), consultar su estado y aprobar las tareas críticas.

Ningún endpoint espera al grafo: solo leen o escriben Redis (cliente async) y encolan. Por eso
un POST tarda milisegundos aunque la corrida del grafo tarde minutos.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Response, status

from app.api.deps import RecursosDep
from app.api.schemas import (
    PATRON_JOB_ID,
    AprobacionAceptada,
    ErrorOut,
    Links,
    TareaAceptada,
    TareaIn,
)
from app.worker.jobs import ConversacionOcupada
from app.worker.models import DecisionHumana, Estado, JobRecord, MensajeReanudar

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
    responses={
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
