"""Piezas compartidas por los especialistas: cómo se invoca a un agente ReAct aislado.

Cada especialista es un `create_react_agent` (su propio mini-grafo modelo <-> herramientas)
que se invoca con UN solo mensaje: la instrucción del supervisor más el contexto mínimo.
No ve `state["messages"]` del orquestador: ni la conversación del usuario, ni las decisiones
del supervisor, ni los tool calls del otro especialista. Lo que devuelve se convierte en una
`Contribution` con autor, tarea, resultado, fuentes (con scores del RAG) y herramientas.
"""

import json
import warnings
from collections.abc import Sequence
from typing import Any

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import RetryPolicy
from langgraph.warnings import LangGraphDeprecatedSinceV10
from pydantic import ValidationError

from app.agents.tool_schemas import SalidaBusqueda
from app.graph.state import Contribution, Especialista

# Tope de pasos del mini-grafo de cada especialista: cada vuelta modelo -> herramientas
# consume 2. Con 9 entran 4 rondas de herramientas más la respuesta final.
SPECIALIST_RECURSION_LIMIT = 9


def es_error_transitorio(exc: Exception) -> bool:
    """429 por cuota POR MINUTO, 5xx (p. ej. 503 "high demand"), timeouts y caídas de red: vale la pena reintentar.

    El 429 por cuota DIARIA (`...PerDay...`) no se libera en minutos: reintentarlo solo gasta
    tiempo, así que se deja pasar (el job queda FAILED con el motivo).
    """
    # El SDK tiene max_retries=0 (los reintentos son de esta política): un timeout de LLM_TIMEOUT_S
    # o un corte de conexión no puede matar el job. APITimeoutError es subclase de APIConnectionError.
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError)):
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "PerDay" not in str(exc)
    return status in {500, 502, 503, 504}


# Reintentos de los nodos que llaman al LLM ante errores transitorios, con esperas largas
# (20 s, 40 s) para que se libere la ventana por minuto del free tier.
RETRY_MODELO = RetryPolicy(max_attempts=3, initial_interval=20.0, backoff_factor=2.0, retry_on=es_error_transitorio)


def crear_agente_react(model: BaseChatModel, tools: Sequence[BaseTool], prompt: str, name: str) -> CompiledStateGraph:
    """`create_react_agent` de LangGraph con sus herramientas acotadas.

    En LangGraph 1.x está marcado como deprecado a favor de `langchain.agents.create_agent` (se
    elimina en la 2.0) pero funciona igual y no suma la dependencia `langchain`; se silencia
    solo ese aviso, solo acá.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LangGraphDeprecatedSinceV10)
        return create_react_agent(model, list(tools), prompt=prompt, name=name)


def armar_prompt(instruccion: str, contexto: str = "") -> str:
    """El único mensaje que recibe un especialista."""
    if not contexto:
        return f"Tarea: {instruccion}"
    return f"Tarea: {instruccion}\n\nContexto (datos que ya obtuvo el equipo):\n{contexto}"


def es_error(msg: ToolMessage) -> bool:
    """Error de validación de args (status="error") o error devuelto como dato por la herramienta."""
    if msg.status == "error":
        return True
    if isinstance(msg.artifact, dict):
        return "error" in msg.artifact
    try:
        data = json.loads(msg.content) if isinstance(msg.content, str) else {}
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and "error" in data


def fuentes_rag_de(tool_messages: list[ToolMessage]) -> list[dict[str, Any]]:
    """Las fuentes del RAG salen del ARTIFACT tipado de `buscar_documentacion` (no del texto).

    Se deduplican por chunk: si dos búsquedas trajeron el mismo, queda el de mayor score.
    """
    por_chunk: dict[str, dict[str, Any]] = {}
    for msg in tool_messages:
        if msg.name != "buscar_documentacion" or not isinstance(msg.artifact, dict) or "error" in msg.artifact:
            continue
        try:
            salida = SalidaBusqueda.model_validate(msg.artifact)
        except ValidationError:
            continue
        for fragmento in salida.resultados:
            fuente = fragmento.model_dump(exclude={"contenido"})
            previa = por_chunk.get(fragmento.chunk_id)
            if previa is None or fuente["score"] > previa["score"]:
                por_chunk[fragmento.chunk_id] = fuente
    return sorted(por_chunk.values(), key=lambda f: f["score"], reverse=True)


async def run_specialist(
    agent: CompiledStateGraph,
    nombre: Especialista,
    paso: int,
    instruccion: str,
    contexto: str = "",
) -> Contribution:
    """Corre el especialista aislado y devuelve su contribución."""
    entrada = [HumanMessage(armar_prompt(instruccion, contexto))]
    try:
        resultado = await agent.ainvoke({"messages": entrada}, config={"recursion_limit": SPECIALIST_RECURSION_LIMIT})
        mensajes = resultado["messages"]
        finales = [m for m in mensajes if isinstance(m, AIMessage) and not m.tool_calls]
        texto = finales[-1].text if finales else ""
        estado = "ok" if texto.strip() else "error"
        if not texto.strip():
            texto = "El especialista no produjo una respuesta final."
    except GraphRecursionError:
        mensajes = entrada
        texto = f"El especialista superó el límite de {SPECIALIST_RECURSION_LIMIT} pasos sin responder."
        estado = "error"

    tool_messages = [m for m in mensajes if isinstance(m, ToolMessage)]
    herramientas = [m.name for m in tool_messages if m.name]
    fuentes_rag = fuentes_rag_de(tool_messages)
    etiquetas = list(dict.fromkeys(f"{f['doc_id']} › {f['seccion']}" for f in fuentes_rag))
    fuentes = etiquetas or sorted({h for h, m in zip(herramientas, tool_messages, strict=False) if not es_error(m)})
    # Si todas las herramientas fallaron, la respuesta no tiene respaldo: se marca como error
    # para que el supervisor lo vea en la rúbrica y decida reintentar o cerrar.
    if tool_messages and all(es_error(m) for m in tool_messages):
        estado = "error"
        fuentes, fuentes_rag = [], []
    return {
        "agente": nombre,
        "paso": paso,
        "tarea": instruccion,
        "resultado": texto,
        "fuentes": fuentes,
        "fuentes_rag": fuentes_rag,
        "herramientas": herramientas,
        "llamadas": [
            {"herramienta": call["name"], "args": call["args"]}
            for m in mensajes
            if isinstance(m, AIMessage)
            for call in m.tool_calls
        ],
        "estado": estado,
    }


def mensaje_de_contribucion(c: Contribution) -> AIMessage:
    """Resumen de la contribución para el historial del orquestador (con autor en `name`)."""
    return AIMessage(content=f"[{c['agente']} | {c['estado']}] {c['resultado']}", name=c["agente"])
