"""Síntesis final: arma la respuesta al usuario a partir de las contribuciones del turno.

Recibe la pregunta, la conversación previa y las contribuciones (con autor y fuentes), no los
mensajes internos de los especialistas. Si el cierre fue forzado por el tope de pasos, o la
última rúbrica del supervisor quedó incompleta, se le avisa para que declare qué faltó en vez
de rellenar con supuestos.

Si el supervisor marcó la pregunta como fuera de dominio, NO se llama al LLM: la respuesta es
un mensaje fijo que explica el alcance del sistema (determinista, sin costo y sin alucinar).

La llamada al modelo es una cadena LCEL (lo del Módulo 2, refactorizado a async):

    ChatPromptTemplate | modelo.with_fallbacks([alternativo]) | StrOutputParser | validar -> RespuestaFinal

- `with_fallbacks`: si el modelo principal falla por un error transitorio (429 de cuota, 5xx,
  timeout), responde `SYNTH_FALLBACK_MODEL`. Con el free tier de Gemini la cuota es por modelo,
  así que el alternativo tiene la suya.
- `with_retry`: una respuesta vacía no se entrega; se reintenta una vez la cadena entera.
- Es el único rol que hace streaming de tokens: el worker los publica y `GET /tasks/{id}/stream`
  los transmite mientras se escribe la respuesta.
"""

import re
from typing import Any

import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

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

PLANTILLA = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT),
        (
            "human",
            "Conversación previa:\n{historial}\n\nPregunta actual del usuario:\n{pregunta}\n\n"
            "Contribuciones:\n{contribuciones}{avisos}",
        ),
    ]
)

# Errores del proveedor que justifican pasar al modelo alternativo (los mismos que se reintentan).
ERRORES_PARA_FALLBACK: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,  # incluye APITimeoutError
    httpx.TransportError,
)
_CITA = re.compile(r"\[([^\[\]]+? › [^\[\]]+?)\]")


class RespuestaVacia(ValueError):
    """El modelo devolvió solo espacios: no se entrega una respuesta vacía."""


class RespuestaFinal(BaseModel):
    """Salida validada de la cadena de síntesis."""

    respuesta: str = Field(min_length=1, description="Texto para el usuario, con las fuentes citadas.")
    citas: list[str] = Field(default_factory=list, description="Fuentes citadas como [doc › sección], sin repetir.")


def validar_respuesta(texto: str) -> RespuestaFinal:
    texto = texto.strip()
    if not texto:
        raise RespuestaVacia("el modelo devolvió una respuesta vacía")
    return RespuestaFinal(respuesta=texto, citas=list(dict.fromkeys(_CITA.findall(texto))))


def build_cadena_sintesis(model: BaseChatModel, fallback: BaseChatModel | None = None) -> Runnable:
    """Cadena LCEL de la síntesis: entrada {historial, pregunta, contribuciones, avisos} -> RespuestaFinal."""
    modelo: Runnable = model
    if fallback is not None:
        modelo = model.with_fallbacks([fallback], exceptions_to_handle=ERRORES_PARA_FALLBACK)
    return (PLANTILLA | modelo | StrOutputParser() | RunnableLambda(validar_respuesta)).with_retry(
        retry_if_exception_type=(RespuestaVacia,), stop_after_attempt=2, wait_exponential_jitter=False
    )


def make_synthesizer_node(model: BaseChatModel, turnos_de_contexto: int = 3, fallback: BaseChatModel | None = None):
    cadena = build_cadena_sintesis(model, fallback)

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
            salida: RespuestaFinal = await cadena.ainvoke(
                {
                    "historial": render_historial(state.get("historial", []), turnos_de_contexto),
                    "pregunta": state["pregunta"],
                    "contribuciones": render_contribuciones(state.get("contributions", [])),
                    "avisos": "\n\n" + "\n".join(avisos) if avisos else "",
                },
                config={"run_name": "cadena_sintesis"},
            )
            texto = salida.respuesta
        return {
            "final_answer": texto,
            "task_completed": True,
            "messages": [AIMessage(content=texto, name="synthesizer")],
        }

    return synthesizer
