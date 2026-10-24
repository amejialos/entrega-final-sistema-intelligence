"""Supervisor: router inteligente y controlador de flujo del orquestador.

En cada turno lee la pregunta, la conversación previa del mismo `thread_id` (los últimos
turnos del `historial`) y las contribuciones acumuladas, y decide con salida estructurada
Pydantic (`SupervisorDecision`, vía `with_structured_output`):
- a quién delega ahora (`researcher` / `analyst`) y con qué instrucción autocontenida, o
- `FINISH`, si la rúbrica de validación se cumple y se puede sintetizar la respuesta, o
- `FINISH` con `fuera_de_dominio=true`, si la pregunta no es sobre Caudal (guardrail: no se
  gasta ni una búsqueda y la síntesis responde con un mensaje fijo, sin LLM).

Antes de terminar aplica una RÚBRICA (cobertura, evidencia, cálculos, consistencia): si algo
falla, devuelve la tarea al especialista que corresponde para que refine su output.

Protección contra el supervisor infinito: `step_count` cuenta sus turnos y, al llegar a
`max_steps`, el nodo NO llama al LLM y fuerza la síntesis (`cierre_forzado=True`).
"""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, model_validator

from app.graph.state import Contribution, Decision, Destino, OrchestratorState, Turno

MAX_STEPS = 6
MAX_CHARS_RESPUESTA_PREVIA = 800

PROMPT = """Sos el Supervisor de un equipo de análisis e investigación sobre la librería de
ingesta de datos **Caudal**. Coordinás a dos especialistas:
- "researcher": Agente Investigador. Busca en la documentación oficial de Caudal
  (base vectorial + búsqueda por palabras clave) y devuelve datos con su fuente. No calcula.
- "analyst": Agente Analista. Hace cálculos con herramientas (aritmética segura, estimación
  de lotes y tiempos) SOLO sobre los datos que le pasa el equipo. No busca información.

Dada la conversación (la pregunta actual, los turnos previos y lo que ya aportó cada
especialista en este turno), decidí ¿quién debe intervenir ahora o es momento de finalizar?

Cómo delegar:
- Elegí UN especialista por turno y dale una `instruccion` autocontenida y específica
  (qué dato buscar o qué cuenta hacer, con los números ya conocidos). El especialista NO ve
  la conversación: solo tu instrucción (y el analista, además, los hallazgos del investigador).
- Primero los datos documentales (researcher), después los cálculos (analyst).
- Seguimientos: si la pregunta se apoya en un turno previo ("¿y con 8 workers?", "¿y eso
  cuánto tarda?"), usá los datos de la conversación previa: pasale al analista en la
  instrucción los números que ya se conocen y no vuelvas a buscar lo que ya está citado.

Fuera de dominio: si la pregunta NO es sobre Caudal, su documentación o un cálculo sobre esos
datos (por ejemplo, deportes, recetas o temas generales), no delegues: next = "FINISH" y
fuera_de_dominio = true.

RÚBRICA DE VALIDACIÓN (evaluala siempre y completá `rubrica`):
1. cobertura: cada parte de la pregunta tiene respuesta en alguna contribución (o en la
   conversación previa, si es un seguimiento).
2. evidencia: cada dato tomado de la documentación tiene una fuente citada por el researcher.
3. calculos: cada número derivado (cuentas, tiempos) lo calculó el analyst con una
   herramienta, usando datos del researcher, de la pregunta o de la conversación previa.
4. consistencia: los datos que usó el analyst coinciden con los del researcher y no hay
   contradicciones entre contribuciones.

Decisión:
- Si los 4 criterios se cumplen -> next = "FINISH".
- Si alguno falla -> delegá al especialista que puede arreglarlo, con una instrucción que
  diga exactamente qué falta o qué hay que corregir (esto es pedir un refinamiento).
- Si un especialista ya devolvió "error" dos veces en lo mismo, no insistas: FINISH y que
  la síntesis declare el faltante.
- Vas por el paso {paso} de {max_steps}. En el último paso disponible, elegí FINISH."""


class Rubrica(BaseModel):
    cobertura: bool = Field(description="Cada parte de la pregunta tiene respuesta.")
    evidencia: bool = Field(description="Cada dato documental tiene fuente citada.")
    calculos: bool = Field(description="Cada número derivado lo calculó el analista con herramienta.")
    consistencia: bool = Field(description="No hay contradicciones entre contribuciones.")

    def cumplida(self) -> bool:
        return all(self.model_dump().values())


class SupervisorDecision(BaseModel):
    """Decisión estructurada del supervisor en un paso (la valida Pydantic al salir del LLM)."""

    rubrica: Rubrica = Field(description="Evaluación de la rúbrica sobre las contribuciones actuales.")
    razon: str = Field(description="Por qué se elige este destino (breve).")
    next: Literal["researcher", "analyst", "FINISH"] = Field(description="Quién interviene ahora, o FINISH.")
    instruccion: str = Field(
        default="",
        description="Instrucción autocontenida para el especialista elegido. Vacía si next es FINISH.",
    )
    fuera_de_dominio: bool = Field(
        default=False, description="true solo si la pregunta no es sobre Caudal (entonces next = FINISH)."
    )

    @model_validator(mode="after")
    def _fuera_de_dominio_cierra(self) -> "SupervisorDecision":
        # Coherencia: una pregunta fuera de dominio nunca se delega.
        if self.fuera_de_dominio:
            self.next = "FINISH"
            self.instruccion = ""
        return self


def render_contribuciones(contributions: list[Contribution]) -> str:
    if not contributions:
        return "(todavía ningún especialista intervino en este turno)"
    bloques = []
    for i, c in enumerate(contributions, start=1):
        bloques.append(
            f"#{i} {c['agente']} (paso {c['paso']}, estado: {c['estado']})\n"
            f"Tarea: {c['tarea']}\n"
            f"Herramientas: {', '.join(c['herramientas']) or 'ninguna'} | Fuentes: {', '.join(c['fuentes']) or 'ninguna'}\n"
            f"Resultado: {c['resultado']}"
        )
    return "\n\n".join(bloques)


def render_historial(historial: list[Turno], ultimos: int = 3) -> str:
    """Los últimos turnos de la conversación (pregunta + respuesta recortada)."""
    if not historial or ultimos <= 0:
        return "(es el primer turno de la conversación)"
    return "\n\n".join(
        f"Turno {t['numero']} - Pregunta: {t['pregunta']}\nRespuesta: {t['respuesta'][:MAX_CHARS_RESPUESTA_PREVIA]}"
        for t in historial[-ultimos:]
    )


def build_decider(model: BaseChatModel, method: str = "json_schema") -> Runnable:
    """El LLM del supervisor con salida estructurada (devuelve un `SupervisorDecision` validado)."""
    return model.with_structured_output(SupervisorDecision, method=method)


def make_supervisor_node(decider: Runnable, max_steps: int = MAX_STEPS, turnos_de_contexto: int = 3):
    async def supervisor(state: OrchestratorState) -> dict[str, Any]:
        paso = state.get("step_count", 0) + 1
        if paso > max_steps:
            # Tope duro: sin llamar al LLM, se cierra con lo que haya.
            razon = f"Se alcanzó el máximo de {max_steps} pasos del supervisor: cierre forzado."
            return {
                "next_agent": "synthesizer",
                "instruccion_actual": "",
                "cierre_forzado": True,
                "decisiones": [_registro(paso, "FINISH", "", razon, {})],
                "messages": [AIMessage(content=f"[supervisor] -> synthesizer: {razon}", name="supervisor")],
            }

        decision: SupervisorDecision = await decider.ainvoke(
            [
                SystemMessage(PROMPT.format(paso=paso, max_steps=max_steps)),
                HumanMessage(
                    f"Conversación previa:\n{render_historial(state.get('historial', []), turnos_de_contexto)}\n\n"
                    f"Pregunta actual del usuario:\n{state['pregunta']}\n\n"
                    f"Contribuciones de este turno:\n{render_contribuciones(state.get('contributions', []))}"
                ),
            ]
        )
        destino: Destino = "synthesizer" if decision.next == "FINISH" else decision.next
        # Un especialista sin instrucción recibiría una tarea vacía: se usa la pregunta.
        instruccion = decision.instruccion.strip() or (state["pregunta"] if destino != "synthesizer" else "")
        return {
            "next_agent": destino,
            "instruccion_actual": instruccion,
            "step_count": paso,
            "fuera_de_dominio": decision.fuera_de_dominio,
            "decisiones": [_registro(paso, decision.next, instruccion, decision.razon, decision.rubrica.model_dump())],
            "messages": [AIMessage(content=f"[supervisor] -> {destino}: {decision.razon}", name="supervisor")],
        }

    return supervisor


def _registro(paso: int, destino: str, instruccion: str, razon: str, rubrica: dict[str, bool]) -> Decision:
    return {"paso": paso, "next": destino, "instruccion": instruccion, "razon": razon, "rubrica": rubrica}


def route_from_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "synthesizer"]:
    """Arista condicional: el `Literal` del retorno define los destinos posibles del grafo."""
    destino = state.get("next_agent", "synthesizer")
    if destino in ("researcher", "analyst"):
        return destino
    return "synthesizer"
