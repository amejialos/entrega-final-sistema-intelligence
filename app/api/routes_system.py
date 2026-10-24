"""Salud del sistema (la usa el healthcheck de Docker Compose)."""

from fastapi import APIRouter, HTTPException, status
from redis.exceptions import RedisError

from app import __version__
from app.api.deps import RecursosDep
from app.api.schemas import ErrorOut, HealthOut, InfoRAG

router = APIRouter(tags=["sistema"])


@router.get(
    "/health",
    summary="Salud del servicio",
    response_model=HealthOut,
    responses={503: {"model": ErrorOut, "description": "Redis no responde."}},
)
async def health(recursos: RecursosDep) -> HealthOut:
    """Redis responde y el grafo y la capa RAG están cargados. No llama a Pinecone ni al LLM."""
    try:
        await recursos.redis.ping()
    except RedisError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Redis no responde.") from exc
    info = recursos.info
    return HealthOut(
        status="ok",
        redis="ok",
        version=__version__,
        modelos=recursos.modelos,
        rag=InfoRAG(
            indice=info.get("indice", "?"),
            namespace=info.get("namespace", "?"),
            chunks_bm25=getattr(recursos.rag, "chunks_indexados", 0),
        ),
        trazas=info.get("trazas", "desactivadas"),
        worker_concurrencia=recursos.worker.concurrencia,
        aprobacion="protegida" if recursos.revisores else "deshabilitada",
        revisores=len(recursos.revisores),
    )
