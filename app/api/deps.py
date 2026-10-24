"""Recursos compartidos de la API (Redis, store, grafo, worker, RAG) y su inyección con `Depends`.

La app no crea nada por su cuenta: recibe un `Recursos` armado por una fábrica (en producción
`app.main.recursos_desde_settings`; en los tests, fakeredis + `InMemorySaver` + LLM fake).
"""

import secrets
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.agents.researcher import Buscador
from app.worker.jobs import JobStore
from app.worker.worker import Worker


@dataclass
class Recursos:
    store: JobStore
    worker: Worker
    redis: Redis
    graph: CompiledStateGraph
    rag: Buscador
    modelos: dict[str, str] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)  # índice, namespace, proyecto de Phoenix...
    clave_revisor: str | None = None  # REVIEWER_API_KEY; None = aprobación abierta


def get_recursos(request: Request) -> Recursos:
    return request.app.state.recursos


RecursosDep = Annotated[Recursos, Depends(get_recursos)]


# `auto_error=False`: sin clave configurada la aprobación queda abierta; el chequeo lo hace `requiere_revisor`.
_header_revisor = APIKeyHeader(
    name="X-Reviewer-Key", auto_error=False, description="Clave de revisor (REVIEWER_API_KEY) para aprobar."
)


async def requiere_revisor(recursos: RecursosDep, clave: Annotated[str | None, Security(_header_revisor)] = None) -> None:
    """Solo quien tiene la clave de revisor aprueba o rechaza una tarea crítica.

    Se compara con `secrets.compare_digest` (tiempo constante): el tiempo de respuesta no revela
    cuántos caracteres de la clave se acertaron.
    """
    esperada = recursos.clave_revisor
    if esperada is None:
        return
    if clave is None or not secrets.compare_digest(clave.encode(), esperada.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Falta la clave de revisor o no es válida (header X-Reviewer-Key).",
            headers={"WWW-Authenticate": "APIKey"},
        )
