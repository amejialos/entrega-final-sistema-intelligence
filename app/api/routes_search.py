"""Búsqueda directa en la capa RAG, sin pasar por los agentes (útil para depurar la recuperación).

Es la misma búsqueda que usa la herramienta del investigador: mismo esquema de entrada
(`ConsultaRAG`), misma salida (`ResultadoBusqueda`, con chunk_id y scores) y el mismo span
RETRIEVER en Phoenix. No llama a ningún LLM de chat (solo al modelo de embeddings).
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.deps import RecursosDep
from app.api.schemas import ErrorOut
from app.rag.schemas import ConsultaRAG, ResultadoBusqueda

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


@router.post(
    "/search", summary="Búsqueda directa en el RAG", response_model=ResultadoBusqueda, responses={503: {"model": ErrorOut}}
)
async def buscar(consulta: ConsultaRAG, recursos: RecursosDep) -> ResultadoBusqueda:
    """Top-k fragmentos de la documentación de Caudal (híbrido por defecto: Pinecone + BM25 con RRF)."""
    try:
        return await recursos.rag.buscar(consulta)
    except Exception as exc:
        # El detalle (que puede traer URLs internas o datos del proveedor) va al log, no al cliente.
        logger.exception("POST /search falló")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"La búsqueda no está disponible ahora ({type(exc).__name__})."
        ) from exc
