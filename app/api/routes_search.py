"""Búsqueda directa en la capa RAG, sin pasar por los agentes (útil para depurar la recuperación).

Es la misma búsqueda que usa la herramienta del investigador: mismo esquema de entrada
(`ConsultaRAG`), misma salida (`ResultadoBusqueda`, con chunk_id y scores) y el mismo span
RETRIEVER en Phoenix. No llama a ningún LLM de chat (solo al modelo de embeddings).
"""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import RecursosDep
from app.api.schemas import ErrorOut
from app.rag.schemas import ConsultaRAG, ResultadoBusqueda

router = APIRouter(tags=["rag"])


@router.post(
    "/search", summary="Búsqueda directa en el RAG", response_model=ResultadoBusqueda, responses={503: {"model": ErrorOut}}
)
async def buscar(consulta: ConsultaRAG, recursos: RecursosDep) -> ResultadoBusqueda:
    """Top-k fragmentos de la documentación de Caudal (híbrido por defecto: Pinecone + BM25 con RRF)."""
    try:
        return await recursos.rag.buscar(consulta)
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"La búsqueda falló: {type(exc).__name__}: {exc}"[:300]) from exc
