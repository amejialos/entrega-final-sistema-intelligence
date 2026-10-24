"""Entradas y salidas de las herramientas de los agentes, validadas con Pydantic.

Cada herramienta declara su `args_schema` (lo que el LLM tiene que mandar) y devuelve una
salida tipada. Se usa `response_format="content_and_artifact"` de LangChain:

- `content`: el JSON que lee el modelo (compacto, para no inflar el contexto);
- `artifact`: la salida completa validada (`model_dump()`), que viaja en el `ToolMessage` pero
  NO se le manda al modelo. De ahí el sistema saca las fuentes con sus scores sin parsear texto.

Si el LLM manda argumentos inválidos (una consulta vacía, `tamano_lote=0`), Pydantic los
rechaza ANTES de ejecutar la herramienta y el `ToolNode` de LangGraph le devuelve el error al
modelo como un `ToolMessage` con `status="error"`, para que corrija y reintente.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.rag.schemas import Fragmento


class ErrorHerramienta(BaseModel):
    """Salida de una herramienta que no pudo completar su trabajo (el error vuelve como dato)."""

    error: str
    sugerencia: str


# --- buscar_documentacion (investigador) ---


class ArgsBusqueda(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consulta: str = Field(
        min_length=3,
        max_length=300,
        description="Qué buscar, en español y con los términos técnicos exactos si los conocés. "
        'Ejemplo: "tamaño de lote por defecto tamano_lote".',
    )


class SalidaBusqueda(BaseModel):
    consulta: str
    resultados: list[Fragmento]


# --- calcular (analista) ---


class ArgsCalculo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expresion: str = Field(
        min_length=1,
        max_length=200,
        description='La cuenta a resolver, solo con números y + - * / // % ** ( ). Ejemplo: "2000000 / 500".',
    )


class SalidaCalculo(BaseModel):
    expresion: str
    resultado: int | float


# --- estimar_tiempo_procesamiento (analista) ---


class ParametrosEstimacion(BaseModel):
    """Todo positivo: se valida antes de calcular."""

    model_config = ConfigDict(extra="forbid")

    registros: int = Field(gt=0, description="Cantidad total de registros/eventos a procesar. Ejemplo: 2000000.")
    tamano_lote: int = Field(gt=0, description="Eventos por lote (tamano_lote de Caudal). Ejemplo: 500.")
    lotes_por_segundo_por_worker: float = Field(gt=0, description="Lotes que procesa un worker por segundo.")
    workers: int = Field(default=1, gt=0, le=10_000, description="Workers en paralelo. Por defecto 1.")


class SalidaEstimacion(BaseModel):
    entrada: ParametrosEstimacion
    lotes: int
    lotes_por_segundo_total: float
    segundos: float
    minutos: float
    formula: str
