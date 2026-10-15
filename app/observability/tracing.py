"""Trazas OpenInference -> Arize Phoenix.

`init_tracing()` registra un `TracerProvider` de OpenTelemetry que exporta a Phoenix
(`phoenix.otel.register`) e instrumenta LangChain/LangGraph con OpenInference
(`LangChainInstrumentor`). Desde ese momento cada corrida del grafo es una traza:

    orquestador job=xxxx (ejecutar) (CHAIN)   <- la corrida completa de un turno
    ├── supervisor (CHAIN) -> ChatOpenAICompatible (LLM): modelo, tokens, costo
    ├── researcher (CHAIN) -> agente ReAct -> LLM + buscar_documentacion (TOOL)
    │                                         └── recuperacion_hibrida (RETRIEVER): consulta,
    │                                             documentos, chunk_id y scores (capa RAG)
    ├── analyst (CHAIN) -> LLM + calcular / estimar_tiempo_procesamiento (TOOL)
    ├── synthesizer (CHAIN) -> LLM
    ├── evaluar_criticidad -> aprobacion_humana (el Interrupt queda como evento del span)
    └── cerrar_turno

Phoenix calcula el COSTO de cada span LLM con los tokens y su tabla de precios por modelo; la
LATENCIA de cada traza es la del span raíz. OpenInference usa el `thread_id` de la metadata
como `session.id`: todos los turnos de una conversación quedan en la misma sesión.

`POST /search` también deja una traza propia (un span RETRIEVER suelto): la capa RAG se puede
observar sin pasar por los agentes.
"""

import logging

from opentelemetry.sdk.trace import TracerProvider

from app.core.config import Settings

logger = logging.getLogger(__name__)


def init_tracing(settings: Settings) -> TracerProvider | None:
    """Registra el exportador a Phoenix e instrumenta LangChain. Devuelve el provider (o None)."""
    if not settings.tracing_enabled:
        logger.info("Trazas desactivadas (TRACING_ENABLED=0)")
        return None
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    provider = register(
        endpoint=settings.phoenix_collector_endpoint,
        project_name=settings.phoenix_project_name,
        batch=True,  # los spans se exportan en segundo plano, sin sumar latencia a cada paso
        set_global_tracer_provider=True,
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    logger.info(
        "Trazas OpenInference -> Phoenix (%s, proyecto %s)", settings.phoenix_collector_endpoint, settings.phoenix_project_name
    )
    return provider


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Exporta los spans pendientes y desinstrumenta (al apagar la API)."""
    if provider is None:
        return
    from openinference.instrumentation.langchain import LangChainInstrumentor

    provider.force_flush()
    LangChainInstrumentor().uninstrument()
    provider.shutdown()
