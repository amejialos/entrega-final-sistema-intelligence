"""Configuración del sistema: TODA sale de variables de entorno, validada con `pydantic-settings`.

Es el único lugar del código que lee el entorno. El resto recibe un `Settings` (o los objetos
que se arman con él) por parámetro, así los tests usan valores propios sin tocar el `.env`.

- Lo que identifica recursos externos NO tiene valor por defecto en el código: el índice y el
  namespace de Pinecone, el modelo de chat y el de embeddings. Si faltan, la API no arranca y
  el error dice qué variable completar (nada de IDs ni modelos hard-codeados).
- Lo que es un ajuste operativo sí tiene un default razonable (URL de Redis local, concurrencia
  del worker, top-k del RAG...), siempre sobreescribible por entorno.
- Las keys son `SecretStr`: no aparecen en logs, reprs ni tracebacks.

Orden de prioridad (el de pydantic-settings): variable de entorno real > `.env` > default.
Por eso el `docker-compose.yml` puede apuntar `REDIS_URL` a `redis://redis:6379/0` aunque el
`.env` diga `localhost`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.rag.schemas import MAX_K

Rol = Literal["supervisor", "research", "analyst", "synth"]


class PrecioModelo(BaseModel):
    """Precio de referencia de un modelo, en US$ por millón de tokens."""

    entrada_usd_por_millon: float = Field(ge=0)
    salida_usd_por_millon: float = Field(ge=0)


# Precios de referencia para estimar el costo por respuesta (US$ / 1M tokens). Son los de la
# tabla de modelos de Arize Phoenix 20.21.1 (consultada por su API GraphQL, `generativeModels`),
# así el costo que devuelve la API coincide con el del dashboard. Se pueden reemplazar con
# PRECIOS_LLM (JSON). Ojo: con el free tier de Gemini el gasto real es 0; esto estima lo que
# costaría pagando. Los embeddings no se cuentan (Phoenix tampoco los costea).
PRECIOS_REFERENCIA: dict[str, PrecioModelo] = {
    "gemini-flash-lite-latest": PrecioModelo(entrada_usd_por_millon=0.30, salida_usd_por_millon=2.50),
    "gemini-3.5-flash-lite": PrecioModelo(entrada_usd_por_millon=0.30, salida_usd_por_millon=2.50),
    "gemini-3.1-flash-lite": PrecioModelo(entrada_usd_por_millon=0.25, salida_usd_por_millon=1.50),
    "gemini-flash-latest": PrecioModelo(entrada_usd_por_millon=0.75, salida_usd_por_millon=3.75),
    "gpt-4o-mini": PrecioModelo(entrada_usd_por_millon=0.15, salida_usd_por_millon=0.60),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # el .env puede tener variables de otras herramientas
        env_ignore_empty=True,  # `OPENAI_BASE_URL=` vacío cuenta como "no definida"
    )

    # --- Modelo de chat (cualquier endpoint compatible con la API de OpenAI) ---
    openai_api_key: SecretStr = Field(description="Key de OpenAI o de Gemini (endpoint compatible).")
    openai_base_url: str | None = Field(default=None, description="Endpoint compatible; vacío = OpenAI.")
    openai_model: str = Field(min_length=1, description="Modelo por defecto de todos los roles.")
    supervisor_model: str | None = None
    research_model: str | None = None
    analyst_model: str | None = None
    synth_model: str | None = None
    supervisor_structured_method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema"
    llm_rpm: float | None = Field(default=None, gt=0, description="Pedidos por minuto POR MODELO (free tier).")
    llm_timeout_s: float = Field(default=90, gt=0)
    precios_llm: dict[str, PrecioModelo] = Field(default_factory=lambda: dict(PRECIOS_REFERENCIA))

    # --- Embeddings y Pinecone (RAG) ---
    embedding_model: str = Field(min_length=1, description="Modelo de embeddings (el mismo de la ingesta).")
    embedding_dimension: int = Field(default=1536, gt=0)
    pinecone_api_key: SecretStr
    index_name: str = Field(min_length=1, description="Índice serverless de Pinecone.")
    pinecone_namespace: str = Field(min_length=1, description="Namespace con los chunks del corpus.")
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    rag_top_k: int = Field(default=5, ge=1, le=MAX_K, description="Fragmentos por búsqueda del investigador.")
    rag_peso_vector: float = Field(default=0.5, ge=0, le=1, description="Peso del vector en la fusión RRF.")
    rag_auto_ingesta: bool = Field(
        default=True, description="Al arrancar, crea el índice y sube el corpus si falta (máquina limpia)."
    )

    # --- Redis: jobs, cola del worker y checkpoints de LangGraph ---
    redis_url: str = "redis://localhost:6379/0"
    redis_prefijo: str = Field(default="final", pattern=r"^[a-z0-9_-]+$")
    job_ttl_s: int = Field(default=7 * 24 * 3600, gt=0)

    # --- Worker y grafo ---
    worker_concurrency: int = Field(default=5, ge=1, le=50)
    worker_embebido: bool = True
    max_steps: int = Field(default=6, ge=1, le=20)
    turnos_de_contexto: int = Field(default=3, ge=0, le=20, description="Turnos previos que ve el supervisor.")

    # --- Observabilidad (Arize Phoenix vía OpenInference) ---
    tracing_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "entrega-final"

    # --- Seguridad: quién puede aprobar las tareas críticas (human-in-the-loop) ---
    reviewer_api_key: SecretStr | None = Field(
        default=None, description="Si está definida, POST /tasks/{id}/approve exige el header X-Reviewer-Key."
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _modelos_por_rol(self) -> "Settings":
        # Un string con espacios sería un modelo inexistente que recién fallaría en el primer pedido.
        for campo in ("openai_model", "supervisor_model", "research_model", "analyst_model", "synth_model"):
            valor = getattr(self, campo)
            if valor is not None and valor != valor.strip():
                setattr(self, campo, valor.strip())
        if self.reviewer_api_key is not None and len(self.reviewer_api_key.get_secret_value()) < 16:
            raise ValueError("REVIEWER_API_KEY tiene que tener al menos 16 caracteres (o no definirla).")
        return self

    def modelo_para(self, rol: Rol) -> str:
        """Modelo de un rol: `<ROL>_MODEL` si está definido; si no, `OPENAI_MODEL`."""
        por_rol = {
            "supervisor": self.supervisor_model,
            "research": self.research_model,
            "analyst": self.analyst_model,
            "synth": self.synth_model,
        }[rol]
        return por_rol or self.openai_model

    def modelos(self) -> dict[str, str]:
        return {rol: self.modelo_para(rol) for rol in ("supervisor", "research", "analyst", "synth")}


@lru_cache
def get_settings() -> Settings:
    """Settings del proceso (se lee el entorno una sola vez)."""
    return Settings()  # type: ignore[call-arg]  # los campos obligatorios vienen del entorno
