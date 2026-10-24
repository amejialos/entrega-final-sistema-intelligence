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
    revisores: dict[str, str] = field(default_factory=dict)  # REVIEWER_KEYS: nombre -> clave; vacío = deshabilitada


def get_recursos(request: Request) -> Recursos:
    return request.app.state.recursos


RecursosDep = Annotated[Recursos, Depends(get_recursos)]


# `auto_error=False`: la falta de clave la responde `requiere_revisor` con 401 y un mensaje claro.
_header_revisor = APIKeyHeader(
    name="X-Reviewer-Key", auto_error=False, description="Clave de revisor (una de REVIEWER_KEYS) para aprobar."
)


async def requiere_revisor(recursos: RecursosDep, clave: Annotated[str | None, Security(_header_revisor)] = None) -> str:
    """Devuelve el nombre del revisor dueño de la clave. Seguro por defecto: sin REVIEWER_KEYS nadie aprueba.

    Se compara contra TODAS las claves con `secrets.compare_digest` (tiempo constante): el tiempo de
    respuesta no revela cuántos caracteres se acertaron ni qué revisor existe.
    """
    if not recursos.revisores:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "La aprobación está deshabilitada: no hay revisores configurados (REVIEWER_KEYS).",
        )
    encontrado = None
    for nombre, esperada in recursos.revisores.items():
        if secrets.compare_digest((clave or "").encode(), esperada.encode()):
            encontrado = nombre
    if not clave or encontrado is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Falta la clave de revisor o no es válida (header X-Reviewer-Key).",
            headers={"WWW-Authenticate": "APIKey"},
        )
    return encontrado


RevisorDep = Annotated[str, Depends(requiere_revisor)]
