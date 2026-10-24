"""Recuperador híbrido ASÍNCRONO: Pinecone (semántico) + BM25 (léxico), fusionados con RRF.

Evoluciona el `RAGSystem` de la Pre-entrega 4. Cambios para la entrega final:

1. **Todo async, sin hilos.** El SDK de Pinecone 10 trae un cliente asíncrono nativo sobre
   httpx (`AsyncPinecone` / `AsyncIndex`) y `OpenAIEmbeddings.aembed_query` usa `AsyncOpenAI`.
   La vía vectorial y la léxica corren a la vez con `asyncio.gather`. BM25 es cálculo en
   memoria sobre 21 chunks (microsegundos): no justifica un hilo.
2. **La fusión es explícita y devuelve los puntajes.** `fusionar_rrf` implementa Reciprocal
   Rank Fusion: cada chunk suma `peso / (c + posición)` por cada lista en la que aparece. Así un
   chunk que está arriba en las dos gana, y uno que solo encuentra BM25 (un código exacto como
   `CAU-417`) igual entra. Cada resultado conserva su score RRF, la similitud coseno de
   Pinecone y el puntaje BM25: son las `fuentes` que devuelve la API.
3. **Es un `BaseRetriever` de LangChain.** Llamarlo con `ainvoke` hace que OpenInference
   registre un span `RETRIEVER` propio en Phoenix, con la consulta, los documentos y sus scores.
"""

import asyncio
import re
import time
import unicodedata
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Protocol

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from app.core.config import Settings
from app.rag.corpus import load_chunks
from app.rag.schemas import ConsultaRAG, Fragmento, Modo, ResultadoBusqueda, Via

RRF_C = 60  # constante clásica de RRF (la misma que usa el EnsembleRetriever de LangChain)
MAX_CANDIDATOS = 20
# Pinecone guarda todos los números de la metadata como float (page=1 vuelve como 1.0).
_CAMPOS_ENTEROS = ("page", "chunk_index")

# Stopwords mínimas en español: BM25 ya penaliza términos frecuentes con el IDF, pero en un
# corpus chico "de", "la" o "que" igual meten ruido en el puntaje.
_STOPWORDS = frozenset(
    "a al algo como con cual cuando de del el en es esa ese eso esta este esto hay la las lo los mas me mi "
    "mis no o para pero por que se si sin sobre su sus tu un una uno unos y ya yo".split()
)
# Conserva palabras con guion o guion bajo como un único token: cau-417, tamano_lote.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize_es(text: str) -> list[str]:
    """Minúsculas, sin tildes, sin stopwords."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return [token for token in _TOKEN.findall(ascii_text) if token not in _STOPWORDS]


ResultadoVia = list[tuple[Document, float]]


class BusquedaVectorial(Protocol):
    async def buscar(self, consulta: str, k: int) -> ResultadoVia: ...


class PineconeBusquedaVectorial:
    """Vía semántica: embedding de la consulta + `query` al índice, las dos cosas async."""

    def __init__(self, index: Any, embeddings: Embeddings, namespace: str) -> None:
        self.index = index  # pinecone.AsyncIndex (o un fake con el mismo `query`)
        self.embeddings = embeddings
        self.namespace = namespace

    async def buscar(self, consulta: str, k: int) -> ResultadoVia:
        vector = await self.embeddings.aembed_query(consulta)
        respuesta = await self.index.query(top_k=k, vector=vector, namespace=self.namespace, include_metadata=True)
        resultados: ResultadoVia = []
        for match in respuesta.matches:
            meta = dict(match.metadata or {})
            texto = meta.pop("text", "")
            for campo in _CAMPOS_ENTEROS:
                if isinstance(meta.get(campo), float):
                    meta[campo] = int(meta[campo])
            meta.setdefault("chunk_id", match.id)
            resultados.append((Document(id=match.id, page_content=texto, metadata=meta), float(match.score)))
        return resultados


class IndiceBM25:
    """Vía léxica: BM25 en memoria sobre los MISMOS chunks que están en Pinecone."""

    def __init__(self, chunks: Sequence[Document]) -> None:
        if not chunks:
            raise ValueError("BM25 necesita los chunks del corpus y la lista está vacía.")
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize_es(c.page_content) for c in self.chunks])

    def buscar(self, consulta: str, k: int) -> ResultadoVia:
        tokens = tokenize_es(consulta)
        if not tokens:
            return []
        puntajes = self._bm25.get_scores(tokens)
        orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)[:k]
        return [(self.chunks[i], float(puntajes[i])) for i in orden if puntajes[i] > 0]


def fusionar_rrf(por_via: dict[Via, ResultadoVia], pesos: dict[Via, float], k: int, c: int = RRF_C) -> list[Fragmento]:
    """Reciprocal Rank Fusion: score(chunk) = Σ_vía peso_vía / (c + posición_en_la_vía)."""
    acumulado: dict[str, dict[str, Any]] = {}
    for via, resultados in por_via.items():
        for posicion, (doc, score) in enumerate(resultados, start=1):
            chunk_id = doc.metadata.get("chunk_id") or doc.id or doc.page_content[:40]
            entrada = acumulado.setdefault(chunk_id, {"doc": doc, "rrf": 0.0, "vector": None, "bm25": None, "vias": []})
            entrada["rrf"] += pesos[via] / (c + posicion)
            entrada[via] = score
            entrada["vias"].append(via)
    solo_una_via = len(por_via) == 1
    fragmentos = []
    for chunk_id, e in acumulado.items():
        meta = e["doc"].metadata
        # Con una sola vía, el score que se reporta es el propio de esa vía (coseno o BM25).
        score = (e["vector"] if e["vector"] is not None else e["bm25"]) if solo_una_via else e["rrf"]
        fragmentos.append(
            Fragmento(
                doc_id=str(meta.get("doc_id", meta.get("source", "?"))),
                seccion=str(meta.get("section", "?")),
                chunk_id=chunk_id,
                titulo=str(meta.get("title", "")),
                score=round(score, 6),
                score_vector=None if e["vector"] is None else round(e["vector"], 6),
                score_bm25=None if e["bm25"] is None else round(e["bm25"], 6),
                recuperado_por=e["vias"],
                contenido=e["doc"].page_content,
            )
        )
    return sorted(fragmentos, key=lambda f: f.score, reverse=True)[:k]


class RecuperadorHibrido(BaseRetriever):
    """`BaseRetriever` async: con `ainvoke(consulta, k=..., modo=...)` deja un span RETRIEVER en Phoenix."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorial: Any = Field(description="Implementa BusquedaVectorial (Pinecone o un fake).")
    bm25: IndiceBM25
    k: int = 5
    peso_vector: float = 0.5
    rrf_c: int = RRF_C

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, k: int | None = None, modo: Modo = "hibrido"
    ) -> list[Document]:
        raise NotImplementedError("El recuperador es asíncrono: usá `await recuperador.ainvoke(...)`.")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        k: int | None = None,
        modo: Modo = "hibrido",
    ) -> list[Document]:
        k = k or self.k
        candidatos = min(MAX_CANDIDATOS, 2 * k) if modo == "hibrido" else k

        async def via_bm25() -> ResultadoVia:
            return self.bm25.buscar(query, candidatos)

        vias: dict[Via, Any] = {}
        if modo in ("hibrido", "vector"):
            vias["vector"] = self.vectorial.buscar(query, candidatos)
        if modo in ("hibrido", "bm25"):
            vias["bm25"] = via_bm25()
        resultados = await asyncio.gather(*vias.values())
        por_via = dict(zip(vias.keys(), resultados, strict=True))
        pesos: dict[Via, float] = {"vector": self.peso_vector, "bm25": 1 - self.peso_vector}
        fragmentos = fusionar_rrf(por_via, pesos, k, self.rrf_c)
        # Los scores viajan en la metadata: Phoenix los muestra en el span del retriever.
        return [
            Document(id=f.chunk_id, page_content=f.contenido, metadata=f.model_dump(exclude={"contenido"})) for f in fragmentos
        ]


class ServicioRAG:
    """Fachada de la capa RAG: recibe un `ConsultaRAG` y devuelve un `ResultadoBusqueda` validado."""

    def __init__(self, recuperador: RecuperadorHibrido) -> None:
        self.recuperador = recuperador

    @property
    def chunks_indexados(self) -> int:
        return len(self.recuperador.bm25.chunks)

    async def buscar(self, consulta: ConsultaRAG) -> ResultadoBusqueda:
        inicio = time.perf_counter()
        docs = await self.recuperador.ainvoke(
            consulta.consulta, k=consulta.k, modo=consulta.modo, config={"run_name": "recuperacion_hibrida"}
        )
        return ResultadoBusqueda(
            consulta=consulta.consulta,
            modo=consulta.modo,
            k=consulta.k,
            fragmentos=[Fragmento.model_validate({**d.metadata, "contenido": d.page_content}) for d in docs],
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 1),
        )


@asynccontextmanager
async def servicio_rag_desde_settings(settings: Settings) -> AsyncIterator[ServicioRAG]:
    """Producción: `AsyncPinecone` + índice async + embeddings async + BM25 sobre `data/`.

    Con `RAG_AUTO_INGESTA` (default) primero se asegura de que el índice exista y tenga el corpus:
    en una cuenta de Pinecone nueva lo crea y lo carga, así `docker compose up` alcanza.

    El chunking (lectura de archivos + tiktoken) se hace una vez al arrancar, en un hilo para no
    frenar el event loop mientras la API levanta. Los clientes de Pinecone se cierran al salir.
    """
    from pinecone import AsyncPinecone

    from app.core.providers import build_embeddings
    from app.rag.ingesta import preparar_indice

    chunks = await asyncio.to_thread(load_chunks)
    async with AsyncPinecone(api_key=settings.pinecone_api_key.get_secret_value()) as pc:
        if settings.rag_auto_ingesta:
            await preparar_indice(settings, pc, build_embeddings(settings))
        index = await pc.index(name=settings.index_name)
        async with index:
            recuperador = RecuperadorHibrido(
                vectorial=PineconeBusquedaVectorial(index, build_embeddings(settings), settings.pinecone_namespace),
                bm25=IndiceBM25(chunks),
                k=settings.rag_top_k,
                peso_vector=settings.rag_peso_vector,
            )
            yield ServicioRAG(recuperador)
