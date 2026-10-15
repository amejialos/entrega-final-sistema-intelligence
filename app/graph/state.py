"""Estado del grafo: una CONVERSACIÓN (thread_id) con varios TURNOS (uno por job).

`OrchestratorState` hereda de `MessagesState` (la conversación, con el reducer `add_messages`).
El checkpointer guarda este estado en Redis bajo el `thread_id`; cada `POST /tasks` con el
mismo `thread_id` es un turno nuevo sobre el estado guardado. Por eso hay dos tipos de campos:

Campos de la CONVERSACIÓN (se acumulan entre turnos y sobreviven a reinicios):
- `messages`: todo lo que se dijo, firmado con `name` (reducer `add_messages`).
- `historial`: un `Turno` por pregunta respondida (reducer `operator.add`). Es lo que ve el
  supervisor para entender un seguimiento ("¿y con 8 workers?") y lo que devuelve
  `GET /conversations/{thread_id}`.

Campos del TURNO (se reinician al empezar cada pregunta, ver `entrada_turno`):
- `pregunta`, `job_id`, `publicar`: lo que pidió el cliente en este job.
- `next_agent` / `instruccion_actual`: a quién delega el supervisor y con qué instrucción.
- `contributions` / `decisiones`: con reducer `operator.add` (cada nodo agrega lo suyo sin pisar
  lo de otro). Para reiniciarlos al empezar un turno se usa `Overwrite([])`, que saltea el
  reducer: es la forma que da LangGraph de "resetear" un canal acumulativo.
- `step_count`, `task_completed`, `cierre_forzado`, `fuera_de_dominio`, `final_answer`.
- `motivos_criticidad`, `aprobacion`, `reporte_publicado`: el human-in-the-loop.

El estado usa TypedDict con valores simples (str, int, dict, list) porque se serializa en cada
checkpoint. Pydantic valida en los BORDES: la decisión del supervisor, los args y salidas de
las herramientas, los mensajes de la cola, los jobs y cada request/response de la API.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Overwrite

Especialista = Literal["researcher", "analyst"]
# Nombres de nodo a los que puede ir el supervisor (la síntesis es el cierre).
Destino = Literal["researcher", "analyst", "synthesizer"]


class Contribution(TypedDict):
    """Una entrega de un especialista: quién, qué se le pidió, qué devolvió y con qué."""

    agente: Especialista
    paso: int  # turno del supervisor que la pidió
    tarea: str  # la instrucción que recibió
    resultado: str  # su respuesta final
    fuentes: list[str]  # etiquetas "doc_id › sección" (investigador) o herramientas (analista)
    fuentes_rag: list[dict[str, Any]]  # `Fuente` del RAG con chunk_id y scores (solo investigador)
    herramientas: list[str]  # herramientas que efectivamente llamó
    llamadas: list[dict[str, Any]]  # cada tool call: {"herramienta", "args"}
    estado: Literal["ok", "error"]


class Decision(TypedDict):
    """Registro de una decisión del supervisor (la versión serializable de `SupervisorDecision`)."""

    paso: int
    next: str
    instruccion: str
    razon: str
    rubrica: dict[str, bool]


class Turno(TypedDict):
    """Una pregunta ya respondida de la conversación."""

    numero: int
    job_id: str
    pregunta: str
    respuesta: str
    fuentes: list[str]
    reporte_publicado: str | None
    fecha: str


class OrchestratorState(MessagesState):
    # conversación
    historial: Annotated[list[Turno], operator.add]
    # turno
    job_id: str
    pregunta: str
    publicar: bool
    next_agent: Destino
    instruccion_actual: str
    contributions: Annotated[list[Contribution], operator.add]
    decisiones: Annotated[list[Decision], operator.add]
    step_count: int
    task_completed: bool
    cierre_forzado: bool
    fuera_de_dominio: bool
    final_answer: str
    motivos_criticidad: list[str]
    aprobacion: dict[str, Any] | None
    reporte_publicado: str | None


def entrada_turno(pregunta: str, job_id: str = "", publicar: bool = False) -> dict[str, Any]:
    """Entrada del grafo para una pregunta nueva (primer turno o seguimiento del mismo hilo).

    `messages` e `historial` no se tocan (se acumulan); todo lo del turno se reinicia.
    """
    return {
        "messages": [HumanMessage(pregunta)],
        "job_id": job_id,
        "pregunta": pregunta,
        "publicar": publicar,
        "next_agent": "synthesizer",
        "instruccion_actual": "",
        "contributions": Overwrite([]),
        "decisiones": Overwrite([]),
        "step_count": 0,
        "task_completed": False,
        "cierre_forzado": False,
        "fuera_de_dominio": False,
        "final_answer": "",
        "motivos_criticidad": [],
        "aprobacion": None,
        "reporte_publicado": None,
    }
