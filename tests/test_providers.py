"""Proveedores: modelo por rol desde Settings, limitador por modelo, thought signatures de Gemini y
qué errores se reintentan."""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.agents.common import es_error_transitorio
from app.core.config import Settings
from app.core.providers import EXTRA_CONTENT_KEY, ChatOpenAICompatible, build_embeddings, build_model

FIRMA = {"google": {"thought_signature": "firma-opaca"}}


def _settings(**extra) -> Settings:
    base = {
        "openai_api_key": "clave-de-prueba",
        "openai_model": "modelo-base",
        "embedding_model": "embedding-x",
        "pinecone_api_key": "p",
        "index_name": "i",
        "pinecone_namespace": "n",
    }
    return Settings(_env_file=None, **{**base, **extra})


def test_modelo_por_rol_con_endpoint_y_sin_reintentos_del_sdk():
    settings = _settings(analyst_model="modelo-analista", openai_base_url="http://localhost:9999/v1")
    model = build_model(settings, "analyst")
    assert isinstance(model, ChatOpenAICompatible)
    assert model.model_name == "modelo-analista"
    assert model.openai_api_base == "http://localhost:9999/v1"
    assert model.temperature == 0 and model.max_retries == 0
    assert build_model(settings, "supervisor").model_name == "modelo-base"


def test_llm_rpm_es_un_limitador_por_modelo():
    settings = _settings(llm_rpm=6, supervisor_model="modelo-a", analyst_model="modelo-a", research_model="modelo-b")
    sup, ana, inv = build_model(settings, "supervisor"), build_model(settings, "analyst"), build_model(settings, "research")
    assert sup.rate_limiter.requests_per_second == pytest.approx(0.1)
    assert sup.rate_limiter is ana.rate_limiter  # mismo modelo: comparten la cuota por minuto
    assert inv.rate_limiter is not sup.rate_limiter  # otro modelo: cuota propia
    assert build_model(_settings(), "synth").rate_limiter is None


def test_embeddings_usan_el_modelo_y_la_dimension_de_la_config():
    emb = build_embeddings(_settings(openai_base_url="http://localhost:9999/v1", embedding_dimension=768))
    assert emb.model == "embedding-x" and emb.dimensions == 768
    assert emb.check_embedding_ctx_length is False  # endpoint compatible: se manda texto, no tokens


def test_conserva_y_reenvia_la_thought_signature():
    model = ChatOpenAICompatible(model="m", api_key="clave-de-prueba")
    respuesta = {
        "id": "r1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calcular", "arguments": '{"expresion": "1+1"}'},
                            "extra_content": FIRMA,
                        }
                    ],
                },
            }
        ],
    }
    mensaje = model._create_chat_result(respuesta).generations[0].message
    assert mensaje.additional_kwargs[EXTRA_CONTENT_KEY] == {"call_1": FIRMA}
    payload = model._get_request_payload([HumanMessage("1+1"), mensaje, ToolMessage("{}", tool_call_id="call_1")])
    assert payload["messages"][1]["tool_calls"][0]["extra_content"] == FIRMA
    assert EXTRA_CONTENT_KEY not in payload["messages"][1]


class _ErrorHTTP(Exception):
    def __init__(self, status_code: int, mensaje: str) -> None:
        super().__init__(mensaje)
        self.status_code = status_code


def test_reintenta_transitorios_pero_no_la_cuota_diaria():
    assert es_error_transitorio(_ErrorHTTP(503, "high demand"))
    assert es_error_transitorio(_ErrorHTTP(429, "quotaId: GenerateRequestsPerMinutePerProjectPerModel"))
    assert not es_error_transitorio(_ErrorHTTP(429, "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert not es_error_transitorio(_ErrorHTTP(400, "bad request"))
