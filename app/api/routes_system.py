"""Salud del sistema (la usa el healthcheck de Docker Compose)."""

from fastapi import APIRouter

from app import __version__
from app.api.deps import RecursosDep
from app.api.schemas import HealthOut, InfoRAG

router = APIRouter(tags=["sistema"])


@router.get("/health", summary="Salud del servicio", response_model=HealthOut)
async def health(recursos: RecursosDep) -> HealthOut:
    """Redis responde y el grafo y la capa RAG están cargados. No llama a Pinecone ni al LLM."""
    await recursos.redis.ping()
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
        aprobacion="protegida" if recursos.clave_revisor else "abierta",
    )
