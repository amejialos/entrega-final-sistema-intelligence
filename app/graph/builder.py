"""El orquestador multi-agente (supervisor + especialistas + síntesis) con HITL y memoria por hilo.

    START -> supervisor --route_from_supervisor--> researcher  -> supervisor
                                              |--> analyst     -> supervisor
                                              |--> synthesizer -> evaluar_criticidad
    evaluar_criticidad --(no crítica)--> cerrar_turno -> END
                       --(crítica)----> aprobacion_humana --interrupt()--> [PAUSA hasta POST /approve]
    aprobacion_humana --(aprobado)--> publicar_reporte -> cerrar_turno
                      --(rechazado)-> cerrar_turno

- **Checkpointer** (`AsyncRedisSaver`): cada super-step queda guardado en Redis bajo el
  `thread_id` de la CONVERSACIÓN. Un job es un turno: el siguiente `POST /tasks` con el mismo
  `thread_id` arranca sobre el estado guardado (con su `historial`). Eso permite seguimientos,
  pausar en `aprobacion_humana` y reanudar desde otro proceso o después de un reinicio.
- **Topología jerárquica**: el supervisor es el único que decide el flujo; los especialistas
  siempre le devuelven el control y nunca se llaman entre sí.
- **`cerrar_turno`** es la única salida a END: agrega el `Turno` al historial de la conversación,
  sea cual sea el camino (respuesta directa, publicada o rechazada).
- **Frenos contra bucles**: `max_steps` en el supervisor (fuerza la síntesis) y `recursion_limit`
  en la invocación (red de seguridad de LangGraph).
- **`RetryPolicy`** en cada nodo que llama a un LLM: reintenta 429 por minuto y 5xx con espera.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from app.agents.analyst import build_analyst_agent, make_analyst_node
from app.agents.common import RETRY_MODELO
from app.agents.researcher import build_research_agent, make_research_node
from app.agents.supervisor import MAX_STEPS, build_decider, make_supervisor_node, route_from_supervisor
from app.agents.synthesizer import make_synthesizer_node
from app.graph.hitl import (
    ACCION_CRITICA,
    Publicador,
    aprobacion_humana,
    evaluar_criticidad,
    make_publicar_node,
    route_criticidad,
)
from app.graph.state import OrchestratorState, Turno

# Cada paso supervisor + especialista son 2 super-steps: 6 pasos = 12, más el que fuerza el
# cierre y la síntesis = 14, más criticidad, aprobación, publicación y cierre = 18. Con 25 sobra
# margen y un bug igual se corta.
RECURSION_LIMIT = 25


async def cerrar_turno(state: OrchestratorState) -> dict[str, Any]:
    """Registra el turno respondido en el historial de la conversación (lo acumula el reducer)."""
    fuentes = sorted({f for c in state.get("contributions", []) if c["agente"] == "researcher" for f in c["fuentes"]})
    turno: Turno = {
        "numero": len(state.get("historial", [])) + 1,
        "job_id": state.get("job_id", ""),
        "pregunta": state["pregunta"],
        "respuesta": state.get("final_answer", ""),
        "fuentes": fuentes,
        "reporte_publicado": state.get("reporte_publicado"),
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return {"historial": [turno]}


def build_graph(
    supervisor_model: BaseChatModel,
    research_model: BaseChatModel,
    analyst_model: BaseChatModel,
    synth_model: BaseChatModel,
    search_tool: BaseTool,
    synth_fallback_model: BaseChatModel | None = None,
    max_steps: int = MAX_STEPS,
    turnos_de_contexto: int = 3,
    structured_method: str = "json_schema",
    retry_policy: RetryPolicy = RETRY_MODELO,
    checkpointer: BaseCheckpointSaver | None = None,
    publicador: Publicador | None = None,
) -> CompiledStateGraph:
    """Arma y compila el orquestador. Modelos, buscador, checkpointer y publicador se inyectan:
    en producción Gemini + Pinecone + Redis; en los tests, fakes e `InMemorySaver`."""
    research_agent = build_research_agent(research_model, search_tool)
    analyst_agent = build_analyst_agent(analyst_model)

    builder = StateGraph(OrchestratorState)
    builder.add_node(
        "supervisor",
        make_supervisor_node(build_decider(supervisor_model, structured_method), max_steps, turnos_de_contexto),
        retry_policy=retry_policy,
    )
    builder.add_node("researcher", make_research_node(research_agent), retry_policy=retry_policy)
    builder.add_node("analyst", make_analyst_node(analyst_agent), retry_policy=retry_policy)
    builder.add_node(
        "synthesizer", make_synthesizer_node(synth_model, turnos_de_contexto, synth_fallback_model), retry_policy=retry_policy
    )
    builder.add_node("evaluar_criticidad", evaluar_criticidad)
    builder.add_node("aprobacion_humana", aprobacion_humana)
    builder.add_node(ACCION_CRITICA, make_publicar_node(publicador or _publicador_no_configurado))
    builder.add_node("cerrar_turno", cerrar_turno)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_from_supervisor)
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("synthesizer", "evaluar_criticidad")
    builder.add_conditional_edges("evaluar_criticidad", route_criticidad)
    # aprobacion_humana decide su destino con Command(goto=...) (lo declara su tipo de retorno).
    builder.add_edge(ACCION_CRITICA, "cerrar_turno")
    builder.add_edge("cerrar_turno", END)
    return builder.compile(name="orquestador", checkpointer=checkpointer)


async def _publicador_no_configurado(job_id: str, reporte: dict[str, Any]) -> str:
    raise RuntimeError("No hay publicador configurado: build_graph(publicador=...).")


def run_config(
    thread_id: str | None = None,
    job_id: str | None = None,
    fase: str | None = None,
    callbacks: Sequence[BaseCallbackHandler] = (),
    recursion_limit: int = RECURSION_LIMIT,
) -> RunnableConfig:
    """Config de una corrida. `thread_id` identifica la conversación en el checkpointer.

    `run_name` y `metadata` viajan a las trazas: en Phoenix cada traza queda nombrada con el job,
    y OpenInference usa el `thread_id` de la metadata como `session.id`, así todos los turnos de
    una conversación (y la reanudación tras una aprobación) quedan en la misma sesión.
    """
    config: RunnableConfig = {"recursion_limit": recursion_limit}
    if callbacks:
        config["callbacks"] = list(callbacks)
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
        etiqueta = f"job={job_id[:8]}" if job_id else f"thread={thread_id[:8]}"
        config["run_name"] = f"orquestador {etiqueta}" + (f" ({fase})" if fase else "")
        config["metadata"] = {
            "thread_id": thread_id,
            **({"job_id": job_id} if job_id else {}),
            **({"fase": fase} if fase else {}),
        }
    return config
