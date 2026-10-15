"""Agente Investigador: la puerta de entrada a la capa RAG.

Herramienta: `buscar_documentacion(consulta)`, búsqueda híbrida (Pinecone + BM25 con RRF)
sobre la documentación de la librería ficticia Caudal (`app/rag/`). Devuelve fragmentos con
su fuente (`doc_id › sección`) para que la respuesta se pueda citar; los scores y el
`chunk_id` viajan en el artifact y terminan en `resultado.fuentes` de la API.

Contexto que recibe: SOLO la instrucción del supervisor. No necesita nada más para buscar.
"""

import json
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from app.agents.common import crear_agente_react, mensaje_de_contribucion, run_specialist
from app.agents.tool_schemas import ArgsBusqueda, ErrorHerramienta, SalidaBusqueda
from app.graph.state import OrchestratorState
from app.rag.schemas import ConsultaRAG, ResultadoBusqueda

NOMBRE = "researcher"
MAX_CHARS_FRAGMENTO = 900  # recorte por fragmento: alcanza para el dato y no infla el contexto

PROMPT = """Sos el Agente Investigador de un equipo de análisis. Tu única fuente es la
documentación oficial de la librería Caudal, que consultás con la herramienta
`buscar_documentacion`.

Reglas:
1. Buscá SIEMPRE antes de responder. Si la primera búsqueda no trae el dato, reformulá
   la consulta (otras palabras, el nombre del parámetro) y buscá de nuevo (máximo 3 búsquedas).
2. Respondé solo con lo que dicen los fragmentos. Citá la fuente de cada dato entre
   corchetes, así: [caudal-instalacion › Configuración]. Si un dato no aparece, decí
   explícitamente "no figura en la documentación"; nunca lo inventes.
3. Si la herramienta devuelve "error", informalo tal cual: no supongas el contenido.
4. No hagas cálculos: tu trabajo es extraer datos. Los cálculos los hace otro agente.
5. Respuesta breve, en español: una lista de datos con su valor numérico exacto y su fuente."""


class Buscador(Protocol):
    """Lo que la herramienta necesita de la capa RAG (el `ServicioRAG` real o un fake)."""

    async def buscar(self, consulta: ConsultaRAG) -> ResultadoBusqueda: ...


def _json(modelo: Any) -> str:
    return json.dumps(modelo, ensure_ascii=False)


def crear_herramienta_busqueda(rag: Buscador, k: int = 5) -> BaseTool:
    """Arma la herramienta sobre la capa RAG inyectada."""

    @tool("buscar_documentacion", args_schema=ArgsBusqueda, response_format="content_and_artifact")
    async def buscar_documentacion(consulta: str) -> tuple[str, dict[str, Any]]:
        """Busca en la documentación oficial de Caudal (búsqueda híbrida semántica + palabras clave).

        Devuelve JSON con hasta 5 fragmentos: {"resultados": [{"fuente", "contenido"}]}.
        Si algo falla devuelve {"error": ..., "sugerencia": ...} en vez de fragmentos.
        """
        try:
            resultado = await rag.buscar(ConsultaRAG(consulta=consulta, k=k))
        except Exception as exc:  # la caída de la fuente vuelve como dato, no rompe el grafo
            error = ErrorHerramienta(
                error=f"la búsqueda falló: {type(exc).__name__}: {exc}"[:300],
                sugerencia="Informá que la documentación no está disponible; no inventes datos.",
            )
            return error.model_dump_json(), error.model_dump()
        if not resultado.fragmentos:
            error = ErrorHerramienta(error="sin resultados", sugerencia="Reformulá la consulta con otros términos.")
            return error.model_dump_json(), error.model_dump()
        salida = SalidaBusqueda(consulta=resultado.consulta, resultados=resultado.fragmentos)
        # Al modelo solo le llega la etiqueta de la fuente y el texto recortado; el artifact
        # conserva chunk_id y scores para la respuesta de la API.
        contenido = _json(
            {"resultados": [{"fuente": f.etiqueta, "contenido": f.contenido[:MAX_CHARS_FRAGMENTO]} for f in salida.resultados]}
        )
        return contenido, salida.model_dump()

    return buscar_documentacion


def build_research_agent(model: BaseChatModel, search_tool: BaseTool) -> CompiledStateGraph:
    """Agente ReAct con una sola herramienta: la búsqueda en la documentación."""
    return crear_agente_react(model, [search_tool], PROMPT, NOMBRE)


def make_research_node(agent: CompiledStateGraph):
    """Nodo del orquestador: corre al investigador con la instrucción del supervisor y nada más."""

    async def researcher(state: OrchestratorState) -> dict[str, Any]:
        contribution = await run_specialist(agent, NOMBRE, paso=state["step_count"], instruccion=state["instruccion_actual"])
        return {"contributions": [contribution], "messages": [mensaje_de_contribucion(contribution)]}

    return researcher
