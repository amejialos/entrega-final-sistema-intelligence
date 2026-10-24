"""Agente Analista (cómputo).

Herramientas, las dos con `args_schema` y salida Pydantic:
- `calcular(expresion)`: aritmética con un evaluador seguro que recorre el AST de Python y solo
  acepta números y operadores (+ - * / // % **). NO usa `eval`: un nombre, una llamada a función
  o un atributo (`__import__("os")`) se rechazan antes de ejecutar nada.
- `estimar_tiempo_procesamiento(...)`: cuántos lotes y cuánto tiempo lleva procesar N registros.
  `ParametrosEstimacion` exige todo > 0: si el modelo manda `tamano_lote=0`, recibe el error de
  validación como dato y corrige.

Contexto que recibe: la instrucción del supervisor + los RESULTADOS del investigador en este
turno (texto y fuentes). Si es un seguimiento y en este turno no se investigó nada, recibe la
respuesta del turno anterior de la conversación, que tiene los datos sobre los que recalcular.
"""

import ast
import math
import operator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from app.agents.common import crear_agente_react, mensaje_de_contribucion, run_specialist
from app.agents.tool_schemas import (
    ArgsCalculo,
    ErrorHerramienta,
    ParametrosEstimacion,
    SalidaCalculo,
    SalidaEstimacion,
)
from app.graph.state import OrchestratorState

NOMBRE = "analyst"
MAX_LARGO_EXPRESION = 200
MAX_EXPONENTE = 100  # 10**100 alcanza; 9**9**9 colgaría el proceso
# Tope de magnitud de cualquier valor intermedio: sobra para lotes, eventos y segundos, y evita
# que una cadena de operaciones "chicas" ((9**99)**99)**99 arme enteros de un millón de dígitos.
MAX_MAGNITUD = 10**30
MAX_CHARS_TURNO_PREVIO = 1500

PROMPT = """Sos el Agente Analista de un equipo de análisis. Hacés cálculos sobre datos que
te pasan; no buscás información ni inventás datos.

Reglas:
1. TODO número que des tiene que salir de una herramienta: `calcular` para aritmética y
   `estimar_tiempo_procesamiento` para lotes y tiempos. No hagas cuentas de cabeza.
2. Usá solo los datos del Contexto o de la Tarea. Si falta un dato necesario, decí cuál
   falta en vez de suponerlo. Si dos datos se contradicen, señalalo explícitamente.
3. Si una herramienta devuelve "error", corregí los argumentos y reintentá una vez.
4. Respuesta breve, en español: cada resultado con la operación que lo produjo y los
   supuestos usados (por ejemplo, el tamaño de lote y de qué fuente salió)."""

_BINARIOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARIOS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class ExpresionInvalida(ValueError):
    pass


def evaluar_expresion(expresion: str) -> float | int:
    """Evalúa aritmética pura recorriendo el AST. Cualquier otro nodo es un error."""
    if len(expresion) > MAX_LARGO_EXPRESION:
        raise ExpresionInvalida(f"la expresión supera {MAX_LARGO_EXPRESION} caracteres")
    try:
        arbol = ast.parse(expresion.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ExpresionInvalida(f"sintaxis inválida: {exc.msg}") from exc

    def _acotado(valor: float | int | complex) -> float | int:
        if isinstance(valor, complex):  # (-8) ** 0.5: Python devuelve un complejo, que no es un dato de Caudal
            raise ExpresionInvalida("el resultado es un número complejo (raíz de un número negativo)")
        if abs(valor) > MAX_MAGNITUD:
            raise ExpresionInvalida(f"resultado demasiado grande (máximo 1e{len(str(MAX_MAGNITUD)) - 1})")
        return valor

    def _eval(nodo: ast.AST) -> float | int:
        if isinstance(nodo, ast.Expression):
            return _eval(nodo.body)
        if isinstance(nodo, ast.Constant) and type(nodo.value) in (int, float):
            return _acotado(nodo.value)
        if isinstance(nodo, ast.BinOp) and type(nodo.op) in _BINARIOS:
            izq, der = _eval(nodo.left), _eval(nodo.right)
            if isinstance(nodo.op, ast.Pow):
                if abs(der) > MAX_EXPONENTE:
                    raise ExpresionInvalida(f"exponente demasiado grande (máximo {MAX_EXPONENTE})")
                # Antes de calcular: |base|**exp > MAX_MAGNITUD se detecta con logaritmos, sin armar el número.
                if abs(izq) > 1 and der > 0 and der * math.log10(abs(izq)) > math.log10(MAX_MAGNITUD):
                    raise ExpresionInvalida("resultado demasiado grande (potencia)")
            return _acotado(_BINARIOS[type(nodo.op)](izq, der))
        if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _UNARIOS:
            return _UNARIOS[type(nodo.op)](_eval(nodo.operand))
        raise ExpresionInvalida(f"elemento no permitido: {type(nodo).__name__} (solo números y + - * / // % **)")

    try:
        return _eval(arbol)
    except ExpresionInvalida:  # es un ValueError: que no la re-envuelva el except de abajo
        raise
    except ZeroDivisionError as exc:
        raise ExpresionInvalida("división por cero") from exc
    except OverflowError as exc:  # float que no entra (1e300 ** 2): un dato para el agente, no un crash
        raise ExpresionInvalida("resultado demasiado grande (overflow)") from exc
    except ValueError as exc:  # p. ej. potencia negativa con exponente fraccionario
        raise ExpresionInvalida(f"operación inválida: {exc}") from exc


@tool("calcular", args_schema=ArgsCalculo, response_format="content_and_artifact")
def calcular(expresion: str) -> tuple[str, dict[str, Any]]:
    """Calcula una expresión aritmética de forma segura.

    Acepta números, paréntesis y los operadores + - * / // % ** (también ^ como potencia).
    No acepta variables, funciones ni texto. Devuelve JSON {"expresion", "resultado"} o
    {"error", "sugerencia"}.
    """
    try:
        resultado = evaluar_expresion(expresion)
    except ExpresionInvalida as exc:
        error = ErrorHerramienta(error=str(exc), sugerencia="Reescribí la cuenta usando solo números y operadores.")
        return error.model_dump_json(), error.model_dump()
    if isinstance(resultado, float) and resultado.is_integer():
        resultado = int(resultado)
    salida = SalidaCalculo(expresion=expresion, resultado=resultado)
    return salida.model_dump_json(), salida.model_dump()


def estimar(params: ParametrosEstimacion) -> SalidaEstimacion:
    lotes = math.ceil(params.registros / params.tamano_lote)
    lotes_por_segundo = params.lotes_por_segundo_por_worker * params.workers
    segundos = lotes / lotes_por_segundo
    return SalidaEstimacion(
        entrada=params,
        lotes=lotes,
        lotes_por_segundo_total=lotes_por_segundo,
        segundos=round(segundos, 2),
        minutos=round(segundos / 60, 2),
        formula="lotes = ceil(registros / tamano_lote); segundos = lotes / (lotes_por_segundo_por_worker * workers)",
    )


@tool("estimar_tiempo_procesamiento", args_schema=ParametrosEstimacion, response_format="content_and_artifact")
def estimar_tiempo_procesamiento(
    registros: int, tamano_lote: int, lotes_por_segundo_por_worker: float, workers: int = 1
) -> tuple[str, dict[str, Any]]:
    """Estima cuántos lotes se forman y cuánto tarda procesar un volumen de registros.

    Devuelve JSON con lotes, segundos, minutos y la fórmula usada. Todos los argumentos tienen
    que ser mayores que 0 (si no, se devuelve el error de validación).
    """
    salida = estimar(
        ParametrosEstimacion(
            registros=registros,
            tamano_lote=tamano_lote,
            lotes_por_segundo_por_worker=lotes_por_segundo_por_worker,
            workers=workers,
        )
    )
    return salida.model_dump_json(), salida.model_dump()


ANALYST_TOOLS: list[BaseTool] = [calcular, estimar_tiempo_procesamiento]


def build_analyst_agent(model: BaseChatModel) -> CompiledStateGraph:
    return crear_agente_react(model, ANALYST_TOOLS, PROMPT, NOMBRE)


def contexto_para_analista(state: OrchestratorState) -> str:
    """Contexto mínimo: los hallazgos del investigador en ESTE turno (con fuentes), no el historial.

    En un seguimiento sin investigación nueva, la respuesta del turno anterior trae los datos.
    """
    hallazgos = [c for c in state.get("contributions", []) if c["agente"] == "researcher" and c["estado"] == "ok"]
    if hallazgos:
        return "\n\n".join(
            f"Hallazgo del investigador (paso {c['paso']}, fuentes: {', '.join(c['fuentes']) or 'sin fuente'}):\n{c['resultado']}"
            for c in hallazgos
        )
    historial = state.get("historial", [])
    if historial:
        previo = historial[-1]
        return (
            f"Respuesta del turno anterior de la conversación (pregunta: {previo['pregunta']}):\n"
            f"{previo['respuesta'][:MAX_CHARS_TURNO_PREVIO]}"
        )
    return ""


def make_analyst_node(agent: CompiledStateGraph):
    async def analyst(state: OrchestratorState) -> dict[str, Any]:
        contribution = await run_specialist(
            agent,
            NOMBRE,
            paso=state["step_count"],
            instruccion=state["instruccion_actual"],
            contexto=contexto_para_analista(state),
        )
        return {"contributions": [contribution], "messages": [mensaje_de_contribucion(contribution)]}

    return analyst
