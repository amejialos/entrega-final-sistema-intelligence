"""Configuración: todo sale del entorno, validado con pydantic-settings; lo que identifica recursos
externos (índice, namespace, modelos) no tiene default en el código."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings

OBLIGATORIAS = ("OPENAI_API_KEY", "OPENAI_MODEL", "EMBEDDING_MODEL", "PINECONE_API_KEY", "INDEX_NAME", "PINECONE_NAMESPACE")
TODAS = (
    *OBLIGATORIAS,
    "OPENAI_BASE_URL",
    "SUPERVISOR_MODEL",
    "RESEARCH_MODEL",
    "ANALYST_MODEL",
    "SYNTH_MODEL",
    "LLM_RPM",
    "PRECIOS_LLM",
    "WORKER_CONCURRENCY",
    "WORKER_EMBEBIDO",
    "REDIS_URL",
    "TRACING_ENABLED",
    "REVIEWER_KEYS",
)


@pytest.fixture
def entorno(monkeypatch):
    for var in TODAS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _completo(entorno) -> None:
    for var in OBLIGATORIAS:
        entorno.setenv(var, f"valor-{var.lower()}")


def test_sin_variables_obligatorias_no_arranca_y_dice_cuales_faltan(entorno):
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    faltantes = {e["loc"][0] for e in error.value.errors() if e["type"] == "missing"}
    assert faltantes == {v.lower() for v in OBLIGATORIAS}


def test_modelo_por_rol_con_fallback_y_vacios_ignorados(entorno):
    _completo(entorno)
    entorno.setenv("ANALYST_MODEL", "modelo-analista")
    entorno.setenv("SUPERVISOR_MODEL", "")  # vacío = no definida
    entorno.setenv("OPENAI_BASE_URL", "")
    settings = Settings(_env_file=None)
    assert settings.modelo_para("analyst") == "modelo-analista"
    assert settings.modelo_para("supervisor") == "valor-openai_model"
    assert settings.openai_base_url is None
    assert settings.index_name == "valor-index_name" and settings.pinecone_namespace == "valor-pinecone_namespace"


def test_las_keys_son_secretas_y_los_tipos_se_validan(entorno):
    _completo(entorno)
    entorno.setenv("LLM_RPM", "4")
    entorno.setenv("WORKER_EMBEBIDO", "0")
    entorno.setenv("PRECIOS_LLM", '{"mi-modelo": {"entrada_usd_por_millon": 0.2, "salida_usd_por_millon": 0.8}}')
    settings = Settings(_env_file=None)
    assert "valor-openai_api_key" not in repr(settings) and "valor-pinecone_api_key" not in str(settings)
    assert settings.openai_api_key.get_secret_value() == "valor-openai_api_key"
    assert settings.llm_rpm == 4.0 and settings.worker_embebido is False
    assert settings.precios_llm["mi-modelo"].salida_usd_por_millon == 0.8

    entorno.setenv("WORKER_CONCURRENCY", "0")
    with pytest.raises(ValidationError, match="worker_concurrency"):
        Settings(_env_file=None)


def test_reviewer_keys_se_leen_como_json_y_validan_nombre_y_largo(entorno):
    _completo(entorno)
    assert Settings(_env_file=None).reviewer_keys == {}  # sin claves: aprobación deshabilitada
    entorno.setenv("REVIEWER_KEYS", '{"ana": "' + "x" * 16 + '"}')
    assert Settings(_env_file=None).reviewer_keys["ana"].get_secret_value() == "x" * 16
    for malo in ('{"ana": "corta"}', '{"Ana Pérez": "' + "x" * 16 + '"}'):
        entorno.setenv("REVIEWER_KEYS", malo)
        with pytest.raises(ValidationError, match="REVIEWER_KEYS"):
            Settings(_env_file=None)


def test_rag_top_k_y_consulta_rag_comparten_el_mismo_limite(entorno):
    from app.rag.schemas import MAX_K, ConsultaRAG

    _completo(entorno)
    entorno.setenv("RAG_TOP_K", str(MAX_K))
    assert ConsultaRAG(consulta="tamaño de lote", k=Settings(_env_file=None).rag_top_k).k == MAX_K
    entorno.setenv("RAG_TOP_K", str(MAX_K + 1))
    with pytest.raises(ValidationError, match="rag_top_k"):
        Settings(_env_file=None)
