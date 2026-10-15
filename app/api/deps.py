"""Recursos compartidos de la API (Redis, store, grafo, worker, RAG) y su inyección con `Depends`.

La app no crea nada por su cuenta: recibe un `Recursos` armado por una fábrica (en producción
`app.main.recursos_desde_settings`; en los tests, fakeredis + `InMemorySaver` + LLM fake).
"""

from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, Request
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


def get_recursos(request: Request) -> Recursos:
    return request.app.state.recursos


RecursosDep = Annotated[Recursos, Depends(get_recursos)]
