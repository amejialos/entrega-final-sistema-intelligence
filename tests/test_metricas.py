"""Métricas por respuesta: el callback cuenta llamadas y tokens por modelo, estima el costo, y el
job devuelve latencia, tokens y costo en `GET /tasks/{id}`."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import PrecioModelo
from app.observability.metricas import Metricas, MetricasLLM, UsoModelo, costo_estimado, normalizar_modelo
from tests.conftest import PREGUNTA, levantar_api, orquestador
from tests.fakes import FakeToolModel


async def test_el_callback_cuenta_tokens_por_modelo_y_estima_el_costo():
    precios = {"Fake-Modelo": PrecioModelo(entrada_usd_por_millon=1.0, salida_usd_por_millon=2.0)}
    contador = MetricasLLM(precios)
    modelo = FakeToolModel(respuestas=[AIMessage("uno"), AIMessage("dos")])
    otro = FakeToolModel(respuestas=[AIMessage("tres")], model_name="sin-precio")
    for m in (modelo, modelo, otro):
        await m.ainvoke([HumanMessage("hola")], config={"callbacks": [contador]})

    metricas = contador.metricas(latencia_ejecucion_s=1.23456)
    assert metricas.llamadas_llm == 3
    assert metricas.por_modelo["fake-modelo"] == UsoModelo(
        llamadas=2, tokens_entrada=200, tokens_salida=40, costo_estimado_usd=0.00028
    )
    assert metricas.tokens_entrada == 300 and metricas.tokens_salida == 60 and metricas.tokens_totales == 360
    assert metricas.costo_estimado_usd == pytest.approx((200 * 1 + 40 * 2) / 1e6)
    assert metricas.modelos_sin_precio == ["sin-precio"]
    assert metricas.latencia_ejecucion_s == 1.235 and metricas.corridas == 1


def test_sumar_acumula_la_corrida_inicial_y_la_reanudacion():
    a = Metricas(
        corridas=1,
        llamadas_llm=5,
        tokens_entrada=500,
        tokens_salida=100,
        tokens_totales=600,
        costo_estimado_usd=0.001,
        latencia_ejecucion_s=2.0,
        por_modelo={"m": UsoModelo(llamadas=5, tokens_entrada=500, tokens_salida=100, costo_estimado_usd=0.001)},
    )
    b = Metricas(corridas=1, latencia_ejecucion_s=0.5, latencia_total_s=30.0)
    total = a.sumar(b)
    assert total.corridas == 2 and total.llamadas_llm == 5 and total.latencia_ejecucion_s == 2.5
    assert total.latencia_total_s == 30.0 and total.por_modelo["m"].llamadas == 5


def test_normalizacion_y_formula_del_costo():
    assert normalizar_modelo(" models/Gemini-3.1-Flash-Lite ") == "gemini-3.1-flash-lite"
    precio = PrecioModelo(entrada_usd_por_millon=0.10, salida_usd_por_millon=0.40)
    assert costo_estimado(1_000_000, 1_000_000, precio) == pytest.approx(0.50)


async def test_el_job_devuelve_latencia_tokens_y_costo(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job = await api.esperar(await api.crear(PREGUNTA), "DONE", "FAILED")
    m = job["metricas"]
    # supervisor (2 decisiones) + investigador (tool call + respuesta) + síntesis = 5 llamadas de 100/20 tokens.
    assert m["llamadas_llm"] == 5 and m["tokens_entrada"] == 500 and m["tokens_salida"] == 100 and m["tokens_totales"] == 600
    assert set(m["por_modelo"]) == {"fake-supervisor", "fake-modelo"}
    assert m["por_modelo"]["fake-supervisor"]["llamadas"] == 2
    assert m["costo_estimado_usd"] == pytest.approx((500 * 1 + 100 * 2) / 1e6)
    assert m["latencia_total_s"] >= m["latencia_ejecucion_s"] > 0
    assert m["corridas"] == 1 and m["modelos_sin_precio"] == []
