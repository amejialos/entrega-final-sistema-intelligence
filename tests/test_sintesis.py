"""La síntesis como cadena LCEL (lo del Módulo 2): prompt | modelo con fallback | parser | validación
Pydantic, con reintento si la respuesta sale vacía."""

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from app.agents.synthesizer import RespuestaFinal, RespuestaVacia, build_cadena_sintesis
from tests.fakes import FakeToolModel

ENTRADA = {"historial": "(sin turnos previos)", "pregunta": "¿Tamaño de lote?", "contribuciones": "researcher: 500", "avisos": ""}
TEXTO = (
    "El tamaño por defecto es 500 [caudal-instalacion › Configuración] y con 4 workers tarda 25 s [caudal-rendimiento › Lotes]."
)


class _ModeloQueFalla(FakeToolModel):
    error: Exception

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recibidos.append(list(messages))
        raise self.error


def _rate_limit() -> openai.RateLimitError:
    respuesta = httpx.Response(429, request=httpx.Request("POST", "https://llm.test/v1/chat/completions"))
    return openai.RateLimitError("quota per minute", response=respuesta, body=None)


async def test_la_cadena_es_lcel_y_devuelve_una_respuesta_validada_con_sus_citas():
    modelo = FakeToolModel(respuestas=[AIMessage(TEXTO)])
    cadena = build_cadena_sintesis(modelo)
    assert isinstance(cadena, Runnable)
    salida = await cadena.ainvoke(ENTRADA)
    assert isinstance(salida, RespuestaFinal)
    assert salida.respuesta == TEXTO
    assert salida.citas == ["caudal-instalacion › Configuración", "caudal-rendimiento › Lotes"]
    system, human = modelo.recibidos[0]
    assert "redactor final" in system.content and "researcher: 500" in human.content


async def test_una_respuesta_vacia_se_reintenta_una_vez():
    modelo = FakeToolModel(respuestas=[AIMessage("   "), AIMessage(TEXTO)])
    salida = await build_cadena_sintesis(modelo).ainvoke(ENTRADA)
    assert salida.respuesta == TEXTO and len(modelo.recibidos) == 2


async def test_dos_respuestas_vacias_terminan_en_error_claro():
    modelo = FakeToolModel(respuestas=[AIMessage("")], repetir_ultima=True)
    with pytest.raises(RespuestaVacia):
        await build_cadena_sintesis(modelo).ainvoke(ENTRADA)


async def test_si_el_modelo_principal_agota_la_cuota_responde_el_fallback():
    principal = _ModeloQueFalla(respuestas=[], error=_rate_limit())
    alternativo = FakeToolModel(respuestas=[AIMessage(TEXTO)], model_name="fake-alternativo")
    salida = await build_cadena_sintesis(principal, fallback=alternativo).ainvoke(ENTRADA)
    assert salida.respuesta == TEXTO
    assert len(principal.recibidos) == 1 and len(alternativo.recibidos) == 1


async def test_un_error_que_no_es_transitorio_no_usa_el_fallback():
    principal = _ModeloQueFalla(respuestas=[], error=ValueError("bug"))
    alternativo = FakeToolModel(respuestas=[AIMessage(TEXTO)])
    with pytest.raises(ValueError, match="bug"):
        await build_cadena_sintesis(principal, fallback=alternativo).ainvoke(ENTRADA)
    assert alternativo.recibidos == []
