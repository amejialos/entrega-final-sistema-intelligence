"""Herramientas de los agentes: `args_schema` Pydantic (rechaza args inválidos antes de ejecutar) y
salidas tipadas (`content_and_artifact`)."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.agents.analyst import ExpresionInvalida, build_analyst_agent, calcular, estimar_tiempo_procesamiento, evaluar_expresion
from app.agents.researcher import crear_herramienta_busqueda
from app.agents.tool_schemas import (
    ArgsBusqueda,
    ArgsCalculo,
    ParametrosEstimacion,
    SalidaBusqueda,
    SalidaCalculo,
    SalidaEstimacion,
)
from tests.fakes import FakeBuscador, FakeToolModel, llamada


def _tool_call(nombre: str, args: dict) -> dict:
    return {"name": nombre, "args": args, "id": "call-1", "type": "tool_call"}


# --- calcular: evaluador seguro por AST ---


@pytest.mark.parametrize(
    ("expresion", "esperado"),
    [("2000000 / 500", 4000), ("(2 + 3) * 4", 20), ("2 ^ 10", 1024), ("-7 // 2", -4), ("10 % 3", 1), ("1.5 * 3", 4.5)],
)
def test_calcular_aritmetica(expresion, esperado):
    assert json.loads(calcular.invoke({"expresion": expresion}))["resultado"] == esperado


@pytest.mark.parametrize(
    "maliciosa",
    [
        "__import__('os').system('echo hackeado')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "x + 1",
        "lambda: 1",
        "[1, 2, 3]",
        "'a' * 3",
    ],
)
def test_calcular_rechaza_codigo(maliciosa):
    salida = json.loads(calcular.invoke({"expresion": maliciosa}))
    assert "error" in salida and "resultado" not in salida


def test_calcular_frena_potencias_gigantes_y_division_por_cero():
    with pytest.raises(ExpresionInvalida, match="exponente"):
        evaluar_expresion("9 ** 9 ** 9")
    assert "división por cero" in json.loads(calcular.invoke({"expresion": "1 / 0"}))["error"]


def test_calcular_devuelve_un_artifact_tipado():
    mensaje = calcular.invoke(_tool_call("calcular", {"expresion": "2000000 / 500"}))
    assert SalidaCalculo.model_validate(mensaje.artifact) == SalidaCalculo(expresion="2000000 / 500", resultado=4000)
    assert json.loads(mensaje.content) == {"expresion": "2000000 / 500", "resultado": 4000}


# --- args_schema: Pydantic valida ANTES de ejecutar ---


@pytest.mark.parametrize(
    ("herramienta", "args"),
    [
        (calcular, {"expresion": ""}),
        (calcular, {"expresion": "1+" * 150}),  # más de 200 caracteres
        (calcular, {"expresion": "1+1", "extra": True}),
        (estimar_tiempo_procesamiento, {"registros": 1000, "tamano_lote": 0, "lotes_por_segundo_por_worker": 5}),
        (estimar_tiempo_procesamiento, {"registros": -1, "tamano_lote": 500, "lotes_por_segundo_por_worker": 5}),
        (estimar_tiempo_procesamiento, {"registros": 1000, "tamano_lote": 500}),  # falta un argumento
    ],
)
def test_args_invalidos_los_rechaza_el_esquema(herramienta, args):
    with pytest.raises(ValidationError):
        herramienta.invoke(args)


async def test_el_agente_recibe_el_error_de_validacion_como_dato_y_corrige():
    """Dentro del agente ReAct el error no rompe nada: el ToolNode lo devuelve como ToolMessage(status="error")."""
    modelo = FakeToolModel(
        respuestas=[
            llamada("estimar_tiempo_procesamiento", {"registros": 1000, "tamano_lote": 0, "lotes_por_segundo_por_worker": 5}),
            llamada("estimar_tiempo_procesamiento", {"registros": 1000, "tamano_lote": 500, "lotes_por_segundo_por_worker": 5}),
            AIMessage("1000 eventos son 2 lotes."),
        ]
    )
    agente = build_analyst_agent(modelo)
    resultado = await agente.ainvoke({"messages": [HumanMessage("Tarea: estimá 1000 eventos")]})
    errores = [m for m in resultado["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert len(errores) == 1 and "tamano_lote" in errores[0].content
    # El modelo vio el error en su segunda llamada y reintentó con argumentos válidos.
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in modelo.recibidos[1])
    assert resultado["messages"][-1].content == "1000 eventos son 2 lotes."


def test_los_esquemas_de_args_son_los_que_ve_el_llm():
    assert calcular.args_schema is ArgsCalculo
    assert estimar_tiempo_procesamiento.args_schema is ParametrosEstimacion
    esquema = estimar_tiempo_procesamiento.tool_call_schema.model_json_schema()
    assert esquema["properties"]["tamano_lote"]["exclusiveMinimum"] == 0
    assert set(esquema["required"]) == {"registros", "tamano_lote", "lotes_por_segundo_por_worker"}


# --- estimar_tiempo_procesamiento ---


def test_estimar_tiempo_calcula_lotes_y_segundos():
    mensaje = estimar_tiempo_procesamiento.invoke(
        _tool_call(
            "estimar_tiempo_procesamiento",
            {"registros": 2_000_000, "tamano_lote": 500, "lotes_por_segundo_por_worker": 5, "workers": 4},
        )
    )
    salida = SalidaEstimacion.model_validate(mensaje.artifact)
    assert salida.lotes == 4000 and salida.segundos == 200 and salida.minutos == pytest.approx(3.33)
    assert salida.entrada.workers == 4


def test_estimar_tiempo_redondea_lotes_para_arriba():
    salida = json.loads(
        estimar_tiempo_procesamiento.invoke({"registros": 1001, "tamano_lote": 500, "lotes_por_segundo_por_worker": 1})
    )
    assert salida["lotes"] == 3


# --- buscar_documentacion (puerta de entrada al RAG) ---


async def test_buscar_devuelve_fragmentos_con_fuente_y_artifact_con_scores():
    buscador = FakeBuscador()
    herramienta = crear_herramienta_busqueda(buscador, k=4)
    mensaje = await herramienta.ainvoke(_tool_call("buscar_documentacion", {"consulta": "tamaño de lote por defecto"}))

    contenido = json.loads(mensaje.content)
    assert [r["fuente"] for r in contenido["resultados"]] == [
        "caudal-instalacion › Configuración",
        "caudal-rendimiento › Backpressure y tamaño de lote",
    ]
    assert "score" not in contenido["resultados"][0]  # al modelo no le hacen falta
    salida = SalidaBusqueda.model_validate(mensaje.artifact)
    assert salida.resultados[0].chunk_id == "caudal-instalacion-p1-c0" and salida.resultados[0].score_vector == 0.8
    assert buscador.consultas[0].consulta == "tamaño de lote por defecto" and buscador.consultas[0].k == 4
    assert herramienta.args_schema is ArgsBusqueda


async def test_buscar_devuelve_error_como_dato_si_pinecone_falla():
    herramienta = crear_herramienta_busqueda(FakeBuscador(falla=ConnectionError("Pinecone no responde")))
    mensaje = await herramienta.ainvoke(_tool_call("buscar_documentacion", {"consulta": "tamano_lote"}))
    assert "Pinecone no responde" in mensaje.artifact["error"]
    assert "no inventes" in json.loads(mensaje.content)["sugerencia"]


async def test_buscar_sin_resultados_y_consulta_invalida():
    herramienta = crear_herramienta_busqueda(FakeBuscador(fragmentos=[]))
    mensaje = await herramienta.ainvoke(_tool_call("buscar_documentacion", {"consulta": "algo inexistente"}))
    assert mensaje.artifact["error"] == "sin resultados"
    with pytest.raises(ValidationError):
        await herramienta.ainvoke({"consulta": "  "})


def test_el_modelo_lee_los_chunks_completos_y_un_corte_queda_marcado():
    """Regresión: con el tope de 900 caracteres de la PE7, el dato de CAU-417 (posición 874 de un
    chunk de 1.765) quedaba cortado y el investigador respondía que no figuraba en la documentación."""
    from app.agents.researcher import MARCA_RECORTE, MAX_CHARS_FRAGMENTO, recortar
    from app.rag.corpus import load_chunks

    chunks = load_chunks(chunk_size=2800, chunk_overlap=400, length_function=len)
    assert max(len(c.page_content) for c in chunks) <= MAX_CHARS_FRAGMENTO
    errores = next(c.page_content for c in chunks if "CAU-417" in c.page_content)
    assert "esquema registrado" in recortar(errores)
    assert recortar("x" * (MAX_CHARS_FRAGMENTO + 1)).endswith(MARCA_RECORTE)


@pytest.mark.parametrize(
    "expresion",
    [
        "1e300 ** 2",  # OverflowError de float
        "((9 ** 99) ** 99) ** 99",  # cada exponente es chico pero el resultado tiene ~1 millón de dígitos
        "10 ** 99 * 10 ** 99",  # producto gigante
        "9" * 60,  # literal gigante
    ],
)
def test_calcular_rechaza_resultados_desmesurados_sin_romper_el_job(expresion):
    """Ninguna cuenta puede tirar una excepción no controlada (el job quedaría FAILED) ni colgar el proceso."""
    with pytest.raises(ExpresionInvalida, match="demasiado grande"):
        evaluar_expresion(expresion)
    assert "demasiado grande" in json.loads(calcular.invoke({"expresion": expresion}))["error"]


def test_calcular_acepta_cuentas_grandes_pero_razonables():
    assert evaluar_expresion("2 ** 64") == 18446744073709551616
    assert evaluar_expresion("2000000 * 500 * 3600") == 3_600_000_000_000
