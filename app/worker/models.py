"""Modelos Pydantic del dominio de ejecución: estados, jobs, resultados y mensajes de la cola.

Los comparten el worker (que los escribe en Redis) y la API (que los devuelve como
`response_model`): lo que se guarda y lo que se responde pasa por el mismo esquema. Un job
mal formado en Redis falla al leerlo con un error claro en vez de llegar roto al cliente.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, TypeAdapter

from app.observability.metricas import Metricas
from app.rag.schemas import Fuente


class Estado(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


FINALES = frozenset({Estado.DONE, Estado.FAILED})


class EventoEstado(BaseModel):
    status: Estado
    t: str = Field(description="Momento de la transición (ISO 8601, UTC).")
    por: str | None = Field(default=None, description="Quién la provocó, si no fue el worker (p. ej. 'approve').")


class DecisionHumana(BaseModel):
    """La señal externa que reanuda una tarea pausada (body de `POST /tasks/{id}/approve`)."""

    model_config = ConfigDict(extra="forbid")

    aprobado: StrictBool = Field(description="true reanuda y ejecuta la acción crítica; false la cancela.")
    comentario: str = Field(default="", max_length=1000)
    revisor: str = Field(default="anónimo", min_length=1, max_length=100)


class AprobacionRegistrada(BaseModel):
    """La decisión tal como quedó en el estado del grafo (con su fecha)."""

    aprobado: bool
    comentario: str = ""
    revisor: str = "anónimo"
    fecha: str | None = None


class AprobacionPendiente(BaseModel):
    """Lo que ve el revisor mientras el job está en WAITING_APPROVAL (el payload del `interrupt`)."""

    accion: str
    motivos: list[str]
    borrador: str
    como_responder: str


class ContribucionOut(BaseModel):
    agente: Literal["researcher", "analyst"]
    paso: int
    estado: Literal["ok", "error"]
    tarea: str
    resultado: str
    fuentes: list[str]
    herramientas: list[str]


class ResultadoTarea(BaseModel):
    """La respuesta de un turno, con su trazabilidad (quién aportó qué y de qué fuente)."""

    respuesta: str
    fuentes: list[Fuente] = Field(description="Fragmentos del RAG que respaldan la respuesta, por score.")
    contribuciones: list[ContribucionOut]
    ruta_supervisor: list[str] = Field(description="Destino elegido por el supervisor en cada paso.")
    turno: int = Field(description="Número de turno dentro de la conversación (thread_id).")
    task_completed: bool
    cierre_forzado: bool
    fuera_de_dominio: bool
    motivos_criticidad: list[str]
    aprobacion: AprobacionRegistrada | None = None
    reporte_publicado: str | None = None


class JobRecord(BaseModel):
    """Estado de un job (un turno de una conversación) tal como vive en Redis."""

    job_id: str
    thread_id: str
    status: Estado
    pregunta: str
    publicar: bool = False
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    resultado: ResultadoTarea | None = None
    error: str | None = None
    aprobacion_pendiente: AprobacionPendiente | None = Field(
        default=None, description="Qué tiene que revisar el humano (solo en WAITING_APPROVAL)."
    )
    aprobacion: DecisionHumana | None = None
    metricas: Metricas | None = Field(default=None, description="Latencia, tokens y costo estimado de este job.")
    eventos: list[EventoEstado] = Field(default_factory=list, description="Historial de estados con su hora.")


# --- Mensajes de la cola del worker (se validan al leerlos de Redis) ---


class MensajeEjecutar(BaseModel):
    tipo: Literal["ejecutar"] = "ejecutar"
    job_id: str


class MensajeReanudar(BaseModel):
    tipo: Literal["reanudar"] = "reanudar"
    job_id: str
    decision: DecisionHumana


MensajeCola = Annotated[MensajeEjecutar | MensajeReanudar, Field(discriminator="tipo")]
MENSAJE_COLA: TypeAdapter[MensajeEjecutar | MensajeReanudar] = TypeAdapter(MensajeCola)
