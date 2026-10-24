"""Contratos de la capa RAG (Pydantic): qué entra a una búsqueda y qué sale.

Los usan tres consumidores, siempre con el mismo esquema:
- la herramienta `buscar_documentacion` del agente investigador (args y salida tipados);
- el endpoint `POST /search` (consulta directa al RAG);
- el `resultado.fuentes` de cada respuesta de la API (de dónde salió cada dato).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Modo = Literal["hibrido", "vector", "bm25"]
Via = Literal["vector", "bm25"]


MAX_K = 10  # tope de fragmentos por búsqueda; lo comparten ConsultaRAG y RAG_TOP_K (Settings)


class ConsultaRAG(BaseModel):
    """Una consulta al recuperador."""

    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(min_length=3, max_length=500, examples=["¿Qué significa el error CAU-417?"])
    k: int = Field(default=5, ge=1, le=MAX_K, description="Cantidad de fragmentos a devolver.")
    modo: Modo = Field(default="hibrido", description="hibrido (vector + BM25 con RRF), vector o bm25.")

    @field_validator("consulta")
    @classmethod
    def _sin_espacios_sueltos(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("la consulta no puede ser solo espacios")
        return valor.strip()


class Fuente(BaseModel):
    """De dónde salió un dato: documento, sección, chunk y qué tan relevante fue."""

    doc_id: str = Field(examples=["caudal-errores"])
    seccion: str = Field(examples=["CAU-417: lote rechazado por tamaño"])
    chunk_id: str = Field(examples=["caudal-errores-p3-c0"])
    titulo: str = ""
    score: float = Field(description="Puntaje de la fusión RRF (o el de la vía única en modo vector/bm25).")
    score_vector: float | None = Field(default=None, description="Similitud coseno de Pinecone, si lo encontró esa vía.")
    score_bm25: float | None = Field(default=None, description="Puntaje BM25, si lo encontró esa vía.")
    recuperado_por: list[Via] = Field(default_factory=list)

    @property
    def etiqueta(self) -> str:
        """Cómo se cita en las respuestas: `doc_id › sección`."""
        return f"{self.doc_id} › {self.seccion}"


class Fragmento(Fuente):
    """Una fuente con su texto (lo que lee el modelo)."""

    contenido: str


class ResultadoBusqueda(BaseModel):
    consulta: str
    modo: Modo
    k: int
    fragmentos: list[Fragmento]
    latencia_ms: float = Field(ge=0)
