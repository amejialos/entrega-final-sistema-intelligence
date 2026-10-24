"""Esquemas Pydantic de la API: cada body y cada respuesta tiene su modelo (`response_model`).

Los bodies usan `extra="forbid"`: un campo desconocido es un 422, no algo que se ignora en
silencio. Los modelos del dominio (jobs, resultados, métricas, fuentes del RAG) se reutilizan
de `app.worker.models` y `app.rag.schemas`: lo que se guarda y lo que se responde es lo mismo.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from app.worker.models import Estado

PATRON_THREAD_ID = r"^[A-Za-z0-9_-]{4,64}$"
PATRON_JOB_ID = r"^[0-9a-f]{32}$"


class TareaIn(BaseModel):
    """Body de `POST /tasks`: una pregunta (un turno de conversación)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "pregunta": "Según la documentación de Caudal, ¿qué tamaño de lote se recomienda para ingesta "
                    "masiva y cuánto tarda procesar 2.000.000 de eventos con 4 workers a 5 lotes por segundo?",
                },
                {"pregunta": "¿Y si en vez de 4 workers uso 8?", "thread_id": "demo-conversacion-1"},
                {"pregunta": "¿Qué significa el error CAU-417 y cómo se resuelve?", "publicar": True},
            ]
        },
    )

    pregunta: str = Field(min_length=10, max_length=2000, description="Consulta sobre Caudal (documentación + cálculos).")
    thread_id: str | None = Field(
        default=None,
        pattern=PATRON_THREAD_ID,
        description="Conversación a continuar. Si no se manda, se abre una nueva (y se devuelve su id).",
    )
    publicar: StrictBool = Field(
        default=False,
        description="Publicar el reporte al terminar: acción con efectos secundarios, la tarea pasa a ser "
        "CRÍTICA y espera aprobación humana.",
    )

    @field_validator("pregunta")
    @classmethod
    def _no_vacia(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("la pregunta no puede ser solo espacios")
        return valor.strip()


class Links(BaseModel):
    estado: str
    conversacion: str
    aprobar: str | None = None


class TareaAceptada(BaseModel):
    """Respuesta 202 de `POST /tasks`: el job quedó encolado."""

    job_id: str
    thread_id: str
    status: Estado
    links: Links


class AprobacionAceptada(BaseModel):
    """Respuesta 202 de `POST /tasks/{id}/approve`: la reanudación quedó encolada."""

    job_id: str
    thread_id: str
    status: Estado
    aprobado: bool
    links: Links


class TurnoOut(BaseModel):
    numero: int
    job_id: str
    pregunta: str
    respuesta: str
    fuentes: list[str]
    reporte_publicado: str | None = None
    fecha: str


class ConversacionOut(BaseModel):
    """Historial de una conversación, leído del checkpoint de LangGraph en Redis."""

    thread_id: str
    turnos: list[TurnoOut] = Field(description="Preguntas ya respondidas, en orden.")
    jobs: list[str] = Field(description="Todos los jobs (turnos) del hilo, incluidos los que fallaron o siguen en curso.")
    job_activo: str | None = Field(description="Job que está usando la conversación ahora (None si está libre).")
    pausado_en: list[str] = Field(description="Nodos pendientes del grafo (p. ej. ['aprobacion_humana'] si espera aprobación).")
    mensajes: int = Field(description="Mensajes acumulados en el estado (`messages`, reducer add_messages).")
    checkpoints: int = Field(description="Checkpoints guardados del hilo (aget_state_history, máx. 500).")


class ErrorOut(BaseModel):
    detail: str


class InfoRAG(BaseModel):
    indice: str
    namespace: str
    chunks_bm25: int


class HealthOut(BaseModel):
    status: Literal["ok"]
    redis: Literal["ok"]
    version: str
    modelos: dict[str, str]
    rag: InfoRAG
    trazas: str = Field(description="Proyecto de Phoenix al que van las trazas, o 'desactivadas'.")
    worker_concurrencia: int
    aprobacion: Literal["protegida", "abierta"] = Field(
        description="'protegida' si POST /approve exige X-Reviewer-Key (REVIEWER_API_KEY definida)."
    )
