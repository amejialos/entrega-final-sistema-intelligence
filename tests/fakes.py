"""Fakes para probar el sistema sin red ni keys.

- `FakeToolModel`: modelo de chat que devuelve, en orden, AIMessages (con o sin tool_calls), con
  `usage_metadata` (para las métricas) y guarda lo que recibió (para el aislamiento de contexto).
- `FakeSupervisorModel`: devuelve el JSON de decisiones `SupervisorDecision` guionadas; su
  `with_structured_output` las valida con Pydantic, como haría el LLM real.
- `FakeBuscador`: reemplaza a la capa RAG (o falla, para probar el error como dato).
- `FakeVectorial`: reemplaza a Pinecone dentro del recuperador híbrido real.
"""

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from app.agents.supervisor import Rubrica, SupervisorDecision
from app.rag.schemas import ConsultaRAG, Fragmento, ResultadoBusqueda

USO = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def llamada(nombre: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """AIMessage que pide una herramienta."""
    call_id = call_id or f"call-{uuid.uuid4().hex[:8]}"
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": args, "id": call_id, "type": "tool_call"}])


class FakeToolModel(BaseChatModel):
    """Responde con `respuestas` en orden. Con `repetir_ultima=True` repite la última para siempre."""

    respuestas: list[AIMessage]
    repetir_ultima: bool = False
    model_name: str = "fake-modelo"
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    herramientas_vinculadas: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolModel":
        self.herramientas_vinculadas = [t.name for t in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recibidos.append(list(messages))
        indice = len(self.recibidos) - 1
        if indice >= len(self.respuestas):
            if not self.repetir_ultima:
                raise AssertionError(f"El modelo fake recibió una llamada de más (#{indice + 1}).")
            indice = len(self.respuestas) - 1
        # copia con id único: add_messages REEMPLAZA un mensaje si llega otro con el mismo id
        respuesta = self.respuestas[indice].model_copy(update={"id": f"fake-{uuid.uuid4()}", "usage_metadata": USO})
        return ChatResult(generations=[ChatGeneration(message=respuesta)])


RUBRICA_OK = Rubrica(cobertura=True, evidencia=True, calculos=True, consistencia=True)
RUBRICA_PENDIENTE = Rubrica(cobertura=False, evidencia=False, calculos=False, consistencia=True)


def decision(
    next_: str, instruccion: str = "", razon: str = "fake", rubrica: Rubrica | None = None, fuera_de_dominio: bool = False
) -> SupervisorDecision:
    if rubrica is None:
        rubrica = RUBRICA_OK if next_ == "FINISH" else RUBRICA_PENDIENTE
    return SupervisorDecision(
        next=next_, instruccion=instruccion, razon=razon, rubrica=rubrica, fuera_de_dominio=fuera_de_dominio
    )


class FakeSupervisorModel(BaseChatModel):
    """El "LLM" devuelve el JSON de la próxima decisión; la salida estructurada lo valida con Pydantic."""

    decisiones: list[SupervisorDecision]
    repetir_ultima: bool = False
    model_name: str = "fake-supervisor"
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    metodo: str | None = None

    @property
    def _llm_type(self) -> str:
        return "fake-supervisor"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.recibidos.append(list(messages))
        indice = len(self.recibidos) - 1
        if indice >= len(self.decisiones):
            if not self.repetir_ultima:
                raise AssertionError(f"El supervisor fake recibió una llamada de más (#{indice + 1}).")
            indice = len(self.decisiones) - 1
        mensaje = AIMessage(content=self.decisiones[indice].model_dump_json(), usage_metadata=USO)
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    def with_structured_output(self, schema: Any, *, method: str = "json_schema", **kwargs: Any) -> Runnable:
        assert schema is SupervisorDecision
        self.metodo = method
        return self | RunnableLambda(lambda msg: SupervisorDecision.model_validate_json(msg.content))


def fragmento(doc_id: str, seccion: str, contenido: str, score: float = 0.016, chunk: int = 0) -> Fragmento:
    return Fragmento(
        doc_id=doc_id,
        seccion=seccion,
        chunk_id=f"{doc_id}-p1-c{chunk}",
        titulo=doc_id,
        score=score,
        score_vector=0.8,
        score_bm25=3.2,
        recuperado_por=["vector", "bm25"],
        contenido=contenido,
    )


FRAG_LOTE = fragmento(
    "caudal-instalacion", "Configuración", "tamano_lote (cuántos eventos se procesan juntos, por defecto 500).", 0.0163, 0
)
FRAG_INGESTA = fragmento(
    "caudal-rendimiento",
    "Backpressure y tamaño de lote",
    "Caudal recomienda lotes de 500 eventos para baja latencia y de 5000 para ingesta masiva.",
    0.0159,
    1,
)


class FakeBuscador:
    """Devuelve `fragmentos` para cualquier consulta; con `falla` levanta la excepción indicada."""

    def __init__(self, fragmentos: list[Fragmento] | None = None, falla: Exception | None = None) -> None:
        self.fragmentos = fragmentos if fragmentos is not None else [FRAG_LOTE, FRAG_INGESTA]
        self.falla = falla
        self.consultas: list[ConsultaRAG] = []
        self.chunks_indexados = 21

    async def buscar(self, consulta: ConsultaRAG) -> ResultadoBusqueda:
        self.consultas.append(consulta)
        if self.falla:
            raise self.falla
        return ResultadoBusqueda(
            consulta=consulta.consulta,
            modo=consulta.modo,
            k=consulta.k,
            fragmentos=self.fragmentos[: consulta.k],
            latencia_ms=1.0,
        )


class FakeVectorial:
    """Reemplaza a Pinecone: devuelve (Document, coseno) fijos y registra las consultas."""

    def __init__(self, resultados: list[tuple[Document, float]], falla: Exception | None = None) -> None:
        self.resultados = resultados
        self.falla = falla
        self.consultas: list[tuple[str, int]] = []

    async def buscar(self, consulta: str, k: int) -> list[tuple[Document, float]]:
        self.consultas.append((consulta, k))
        if self.falla:
            raise self.falla
        return self.resultados[:k]
