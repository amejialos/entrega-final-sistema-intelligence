"""Síntesis final: arma la respuesta al usuario a partir de las contribuciones del turno.

Recibe la pregunta, la conversación previa y las contribuciones (con autor y fuentes), no los
mensajes internos de los especialistas. Si el cierre fue forzado por el tope de pasos, o la
última rúbrica del supervisor quedó incompleta, se le avisa para que declare qué faltó en vez
de rellenar con supuestos.

Si el supervisor marcó la pregunta como fuera de dominio, NO se llama al LLM: la respuesta es
un mensaje fijo que explica el alcance del sistema (determinista, sin costo y sin alucinar).
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.supervisor import render_contribuciones, render_historial
from app.graph.state import OrchestratorState

RESPUESTA_FUERA_DE_DOMINIO = (
    "Esta pregunta está fuera del alcance del sistema. Respondo consultas sobre la librería de "
    "ingesta Caudal (instalación, configuración, conectores, rendimiento, seguridad, errores y "
    "changelog) y cálculos sobre esos datos, como lotes y tiempos de procesamiento. No busqué ni "
    "inventé información sobre otros temas."
)

PROMPT = """Sos el redactor final de un equipo de análisis. Escribí la respuesta al usuario
usando EXCLUSIVAMENTE las contribuciones de los especialistas (y, si la pregunta es un
seguimiento, los datos de la conversación previa).

Reglas:
- Respondé cada parte de la pregunta, en español, de forma clara y breve.
- Citá la fuente documental de cada dato entre corchetes (como la dio el investigador) y
  atribuí los cálculos al analista, mostrando la operación.
- Si hay datos contradictorios entre contribuciones, priorizá el dato documental con fuente
  y mencioná la discrepancia.
- Si algo quedó sin resolver, decilo explícitamente. No agregues datos que no estén en las
  contribuciones ni en la conversación previa."""


def make_synthesizer_node(model: BaseChatModel, turnos_de_contexto: int = 3):
    async def synthesizer(state: OrchestratorState) -> dict[str, Any]:
        if state.get("fuera_de_dominio"):
            texto = RESPUESTA_FUERA_DE_DOMINIO
        else:
            avisos = []
            if state.get("cierre_forzado"):
                avisos.append("ATENCIÓN: el cierre fue forzado por el límite de pasos; la tarea puede estar incompleta.")
            decisiones = state.get("decisiones", [])
            rubrica = decisiones[-1]["rubrica"] if decisiones else {}
            incumplidos = [criterio for criterio, ok in rubrica.items() if not ok]
            if incumplidos:
                avisos.append(f"ATENCIÓN: la validación del supervisor no aprobó: {', '.join(incumplidos)}.")
            respuesta = await model.ainvoke(
                [
                    SystemMessage(PROMPT),
                    HumanMessage(
                        f"Conversación previa:\n{render_historial(state.get('historial', []), turnos_de_contexto)}\n\n"
                        f"Pregunta actual del usuario:\n{state['pregunta']}\n\n"
                        f"Contribuciones:\n{render_contribuciones(state.get('contributions', []))}"
                        + ("\n\n" + "\n".join(avisos) if avisos else "")
                    ),
                ]
            )
            texto = respuesta.text
        return {
            "final_answer": texto,
            "task_completed": True,
            "messages": [AIMessage(content=texto, name="synthesizer")],
        }

    return synthesizer
