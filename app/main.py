"""API REST del Sistema Intelligence (FastAPI, asíncrona) y raíz de composición.

    POST /tasks                   crea un turno (PENDING), lo encola y responde 202 + job_id + thread_id
    GET  /tasks/{job_id}          estado: PENDING | RUNNING | WAITING_APPROVAL | DONE | FAILED (+ resultado y métricas)
    GET  /tasks                   últimos jobs
    POST /tasks/{job_id}/approve  decisión humana para una tarea crítica pausada
    GET  /conversations/{id}      historial de la conversación (del checkpoint en Redis)
    POST /search                  búsqueda directa en la capa RAG
    GET  /health                  Redis responde, grafo y RAG cargados

`recursos_desde_settings` arma todo el sistema a partir de `Settings` (variables de entorno):
trazas -> Redis -> checkpointer -> capa RAG (Pinecone async + BM25) -> modelos por rol -> grafo
-> worker. `create_app(fabrica)` recibe la fábrica: los tests le pasan una con fakes.

Levantar:  uv run uvicorn app.main:app --port 8000     (o `docker compose up --build`)
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app import __version__
from app.api import routes_conversations, routes_search, routes_system, routes_tasks
from app.api.deps import Recursos
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def recursos_desde_settings(settings: Settings, iniciar_worker: bool = True) -> AsyncIterator[Recursos]:
    """Producción: Redis real, `AsyncRedisSaver`, Pinecone async + BM25, Gemini/OpenAI y trazas a Phoenix."""
    from app.agents.researcher import crear_herramienta_busqueda
    from app.core.providers import build_model
    from app.graph.builder import build_graph
    from app.graph.checkpointer import redis_checkpointer
    from app.graph.hitl import publicador_redis
    from app.observability.tracing import init_tracing, shutdown_tracing
    from app.rag.retriever import servicio_rag_desde_settings
    from app.worker.jobs import JobStore
    from app.worker.worker import Worker

    provider = init_tracing(settings)  # antes de construir el grafo, para que todo quede instrumentado
    async with AsyncExitStack() as pila:
        pila.callback(shutdown_tracing, provider)
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        pila.push_async_callback(redis.aclose)
        await redis.ping()
        checkpointer = await pila.enter_async_context(redis_checkpointer(settings.redis_url))
        rag = await pila.enter_async_context(servicio_rag_desde_settings(settings))
        graph = build_graph(
            supervisor_model=build_model(settings, "supervisor"),
            research_model=build_model(settings, "research"),
            analyst_model=build_model(settings, "analyst"),
            synth_model=build_model(settings, "synth"),
            search_tool=crear_herramienta_busqueda(rag, k=settings.rag_top_k),
            max_steps=settings.max_steps,
            turnos_de_contexto=settings.turnos_de_contexto,
            structured_method=settings.supervisor_structured_method,
            checkpointer=checkpointer,
            publicador=publicador_redis(redis, settings.redis_prefijo),
        )
        store = JobStore(redis, settings.redis_prefijo, settings.job_ttl_s)
        worker = Worker(redis, store, graph, settings.precios_llm, concurrencia=settings.worker_concurrency)
        if iniciar_worker:
            await worker.iniciar()
            pila.push_async_callback(worker.detener)
        if settings.reviewer_api_key is None:
            logger.warning("REVIEWER_API_KEY no está definida: cualquiera puede aprobar las tareas críticas.")
        logger.info(
            "Sistema listo: modelos %s, índice %s/%s", settings.modelos(), settings.index_name, settings.pinecone_namespace
        )
        yield Recursos(
            store=store,
            worker=worker,
            redis=redis,
            graph=graph,
            rag=rag,
            modelos=settings.modelos(),
            info={
                "indice": settings.index_name,
                "namespace": settings.pinecone_namespace,
                "trazas": settings.phoenix_project_name if provider else "desactivadas",
            },
            clave_revisor=settings.reviewer_api_key.get_secret_value() if settings.reviewer_api_key else None,
        )


FabricaRecursos = Callable[[], AbstractAsyncContextManager[Recursos]]


def _fabrica_por_defecto() -> AbstractAsyncContextManager[Recursos]:
    from app.core.logging import configurar_logging

    settings = get_settings()
    configurar_logging(settings.log_level)
    # WORKER_EMBEBIDO=0: la API solo encola y los jobs los corren procesos `python -m app.worker`.
    return recursos_desde_settings(settings, iniciar_worker=settings.worker_embebido)


DESCRIPCION = """
Sistema de IA de punta a punta sobre la documentación de la librería de ingesta **Caudal**:

- **RAG híbrido** (Pinecone + BM25 con fusión RRF), con las fuentes y sus scores en cada respuesta.
- **Agentes con Supervisor** (LangGraph): investigador, analista con calculadora segura y síntesis.
- **API asíncrona**: `POST /tasks` responde `202` al instante; el grafo corre en un worker con cola en Redis.
- **Conversaciones persistentes** por `thread_id` (checkpointer `AsyncRedisSaver`): sobreviven a reinicios.
- **Human-in-the-loop**: las tareas que publican un reporte esperan `POST /tasks/{id}/approve`.
- **Trazas** en Arize Phoenix (OpenInference) y **métricas por respuesta** (latencia, tokens, costo).
"""

TAGS = [
    {"name": "tareas", "description": "Crear turnos de conversación y consultar su estado (polling)."},
    {"name": "human-in-the-loop", "description": "Aprobar o rechazar las tareas críticas pausadas."},
    {"name": "conversaciones", "description": "Historial de un `thread_id`, leído del checkpoint en Redis."},
    {"name": "rag", "description": "Consulta directa a la capa de recuperación."},
    {"name": "sistema", "description": "Salud del servicio."},
]


def create_app(fabrica: FabricaRecursos = _fabrica_por_defecto) -> FastAPI:
    """Fábrica de la app. Los tests le pasan recursos con fakes (fakeredis, LLM fake, InMemorySaver)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with fabrica() as recursos:
            app.state.recursos = recursos
            yield

    app = FastAPI(
        title="Sistema Intelligence: RAG + agentes con Supervisor",
        version=__version__,
        description=DESCRIPCION,
        openapi_tags=TAGS,
        lifespan=lifespan,
        # FastAPI trae telemetría HTTP nativa que usa el TracerProvider global (el de Phoenix).
        # Se apaga: cada GET de polling sería una traza más y taparía las que importan (las
        # corridas del grafo y las búsquedas), además de distorsionar la latencia del proyecto.
        telemetry={"tracing": False, "metrics": False, "logs": False},
    )
    app.include_router(routes_tasks.router)
    app.include_router(routes_conversations.router)
    app.include_router(routes_search.router)
    app.include_router(routes_system.router)
    return app


app = create_app()
