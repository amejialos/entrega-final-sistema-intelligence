"""Fixtures sin red: fakeredis (jobs, cola y candados), `InMemorySaver` (checkpoints), LLM y RAG fake.

La app se arma con `create_app(fabrica)` y se le habla con httpx por ASGI corriendo su `lifespan`,
igual que en producción pero con los recursos reemplazados.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.agents.researcher import crear_herramienta_busqueda
from app.api.deps import Recursos
from app.core.config import PrecioModelo
from app.graph.builder import build_graph
from app.graph.state import OrchestratorState
from app.main import create_app
from app.worker.jobs import JobStore
from app.worker.worker import Worker
from tests.fakes import FakeBuscador, FakeSupervisorModel, FakeToolModel, decision, llamada

PREGUNTA = "¿Cuál es el tamano_lote por defecto según la documentación de Caudal?"
RESPUESTA = "El tamano_lote por defecto es 500 [caudal-instalacion › Configuración]."
CLAVE_ANA = "clave-de-ana-0123456789"  # clave de revisora de los tests (REVIEWER_KEYS)
# Precios de prueba para los modelos fake: el costo de las métricas es determinista.
PRECIOS_TEST: dict[str, PrecioModelo] = {
    "gemini-flash-lite-latest": PrecioModelo(entrada_usd_por_millon=0.30, salida_usd_por_millon=2.50),
    "fake-supervisor": PrecioModelo(entrada_usd_por_millon=1.0, salida_usd_por_millon=2.0),
    "fake-modelo": PrecioModelo(entrada_usd_por_millon=1.0, salida_usd_por_millon=2.0),
}
SIN_REINTENTOS = RetryPolicy(max_attempts=1)


class PublicadorFake:
    """Registra cada publicación en vez de tener efectos reales."""

    def __init__(self) -> None:
        self.publicados: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, job_id: str, reporte: dict[str, Any]) -> str:
        self.publicados.append((job_id, reporte))
        return f"fake:reporte:{job_id}"


def grafo_simple(compuerta: asyncio.Event | None = None, falla: Exception | None = None):
    """Grafo de un nodo: espera la `compuerta` (simula una corrida larga) o levanta `falla`."""

    async def trabajo(state: OrchestratorState) -> dict[str, Any]:
        if compuerta is not None:
            await compuerta.wait()
        if falla is not None:
            raise falla
        return {"final_answer": f"respuesta a: {state['pregunta']}", "task_completed": True}

    builder = StateGraph(OrchestratorState)
    builder.add_node("trabajo", trabajo)
    builder.add_edge(START, "trabajo")
    builder.add_edge("trabajo", END)
    return builder.compile(checkpointer=InMemorySaver())


def orquestador(
    publicador: PublicadorFake | None = None,
    checkpointer=None,
    decisiones=None,
    investigacion=None,
    analisis=None,
    sintesis=None,
    buscador: FakeBuscador | None = None,
    sin_llm: bool = False,
):
    """El grafo real (supervisor -> investigador -> síntesis por defecto) con modelos fake.

    Con `sin_llm=True` los modelos fallan si se los llama: sirve para probar que una reanudación
    NO repite los pasos que ya estaban guardados en el checkpoint.
    """
    if sin_llm:
        decisiones, investigacion, analisis, sintesis = [], [], [], []
    if decisiones is None:
        decisiones = [decision("researcher", "Buscá tamano_lote."), decision("FINISH")]
    if investigacion is None:
        investigacion = [
            llamada("buscar_documentacion", {"consulta": "tamano_lote"}),
            AIMessage("500 [caudal-instalacion › Configuración]"),
        ]
    if sintesis is None:
        sintesis = [AIMessage(RESPUESTA)]
    return build_graph(
        supervisor_model=FakeSupervisorModel(decisiones=decisiones),
        research_model=FakeToolModel(respuestas=investigacion),
        analyst_model=FakeToolModel(respuestas=analisis or []),
        synth_model=FakeToolModel(respuestas=sintesis),
        search_tool=crear_herramienta_busqueda(buscador or FakeBuscador()),
        retry_policy=SIN_REINTENTOS,
        checkpointer=checkpointer or InMemorySaver(),
        publicador=publicador or PublicadorFake(),
    )


class Entorno:
    def __init__(self, client: httpx.AsyncClient, store: JobStore, worker: Worker, redis: FakeAsyncRedis, app) -> None:
        self.client = client
        self.store = store
        self.worker = worker
        self.redis = redis
        self.app = app

    async def crear(self, pregunta: str = PREGUNTA, **extra: Any) -> str:
        r = await self.client.post("/tasks", json={"pregunta": pregunta, **extra})
        assert r.status_code == 202, r.text
        return r.json()["job_id"]

    async def esperar(self, job_id: str, *estados: str, plazo_s: float = 5.0) -> dict[str, Any]:
        """Polling como haría un cliente real, hasta que el job llegue a uno de `estados`."""
        limite = time.monotonic() + plazo_s
        while True:
            job = (await self.client.get(f"/tasks/{job_id}")).json()
            if job["status"] in estados:
                return job
            if time.monotonic() > limite:
                raise AssertionError(f"El job {job_id} quedó en {job['status']}, esperaba {estados}")
            await asyncio.sleep(0.02)


@asynccontextmanager
async def levantar_api(
    graph,
    rag: FakeBuscador | None = None,
    concurrencia: int = 5,
    redis: FakeAsyncRedis | None = None,
    revisores: dict[str, str] | None = None,
) -> AsyncIterator[Entorno]:
    """La API completa con fakes. Pasando el mismo `redis` (y un grafo con el mismo checkpointer)
    se simula un reinicio: la API nueva ve los jobs y las conversaciones de la anterior."""
    propio = redis is None
    redis = redis or FakeAsyncRedis(decode_responses=True)
    store = JobStore(redis, prefijo="final")
    worker = Worker(redis, store, graph, PRECIOS_TEST, concurrencia=concurrencia, espera_cola_s=0.05)
    rag = rag or FakeBuscador()
    # Por defecto hay una revisora ("ana") y el cliente manda su clave, como haría un revisor real.
    revisores = {"ana": CLAVE_ANA} if revisores is None else revisores

    @asynccontextmanager
    async def fabrica() -> AsyncIterator[Recursos]:
        await worker.iniciar()
        try:
            yield Recursos(
                store=store,
                worker=worker,
                redis=redis,
                graph=graph,
                rag=rag,
                modelos={
                    "supervisor": "fake-supervisor",
                    "research": "fake-modelo",
                    "analyst": "fake-modelo",
                    "synth": "fake-modelo",
                },
                info={"indice": "indice-test", "namespace": "ns-test", "trazas": "desactivadas"},
                revisores=revisores,
            )
        finally:
            await worker.detener()

    app = create_app(fabrica)
    async with app.router.lifespan_context(app):
        transporte = httpx.ASGITransport(app=app)
        headers = {"X-Reviewer-Key": CLAVE_ANA} if "ana" in revisores else {}
        async with httpx.AsyncClient(transport=transporte, base_url="http://test", headers=headers) as client:
            yield Entorno(client, store, worker, redis, app)
    if propio:
        await redis.aclose()


@pytest.fixture
def publicador() -> PublicadorFake:
    return PublicadorFake()
