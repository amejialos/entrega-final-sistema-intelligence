"""Capa RAG: BM25 sobre el corpus real, fusión RRF con sus scores, recuperador async con span
RETRIEVER, `POST /search`, las fuentes en la respuesta de un job y los helpers de la ingesta."""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document

from app.core.config import Settings
from app.rag.corpus import load_chunks
from app.rag.ingesta import IndiceIncompatible, armar_registros, asegurar_indice, preparar_indice
from app.rag.retriever import (
    IndiceBM25,
    PineconeBusquedaVectorial,
    RecuperadorHibrido,
    ServicioRAG,
    fusionar_rrf,
    tokenize_es,
)
from app.rag.schemas import ConsultaRAG, ResultadoBusqueda
from tests.conftest import PREGUNTA, levantar_api, orquestador
from tests.fakes import FakeVectorial


@pytest.fixture(scope="module")
def chunks() -> list[Document]:
    # Chunking por caracteres (sin descargar el tokenizador de tiktoken): ~4 caracteres por token.
    return load_chunks(chunk_size=2800, chunk_overlap=400, length_function=len)


def _doc(chunk_id: str, texto: str = "texto", doc_id: str = "caudal-x", seccion: str = "S") -> Document:
    return Document(
        id=chunk_id, page_content=texto, metadata={"chunk_id": chunk_id, "doc_id": doc_id, "section": seccion, "title": doc_id}
    )


def test_tokenizador_conserva_codigos_y_parametros():
    assert tokenize_es("¿Qué es el error CAU-417 y el tamano_lote?") == ["error", "cau-417", "tamano_lote"]


def test_el_corpus_es_determinista_y_bm25_encuentra_codigos_exactos(chunks):
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)  # ids únicos
    assert load_chunks(chunk_size=2800, chunk_overlap=400, length_function=len)[0].id == chunks[0].id
    top = IndiceBM25(chunks).buscar("CAU-417", 3)
    assert top and top[0][0].metadata["doc_id"] == "caudal-errores"
    assert IndiceBM25(chunks).buscar("de la que", 3) == []  # solo stopwords


def test_fusion_rrf_suma_peso_sobre_c_mas_posicion():
    a, b, c = _doc("a"), _doc("b"), _doc("c")
    fragmentos = fusionar_rrf(
        {"vector": [(a, 0.91), (b, 0.80)], "bm25": [(b, 7.5), (c, 4.0)]}, pesos={"vector": 0.5, "bm25": 0.5}, k=3, c=60
    )
    por_id = {f.chunk_id: f for f in fragmentos}
    assert [f.chunk_id for f in fragmentos] == ["b", "a", "c"]  # b está en las dos listas
    assert por_id["b"].score == pytest.approx(0.5 / 62 + 0.5 / 61, abs=1e-6)
    assert por_id["a"].score == pytest.approx(0.5 / 61, abs=1e-6)
    assert por_id["b"].score_vector == 0.8 and por_id["b"].score_bm25 == 7.5
    assert por_id["b"].recuperado_por == ["vector", "bm25"] and por_id["c"].recuperado_por == ["bm25"]
    assert por_id["c"].score_vector is None


class _CapturaRetriever(AsyncCallbackHandler):
    """Lo que vería OpenInference: el span RETRIEVER con sus documentos."""

    def __init__(self) -> None:
        self.documentos: list[Document] = []

    async def on_retriever_end(self, documents, **kwargs: Any) -> None:
        self.documentos = list(documents)


async def test_recuperador_hibrido_async_y_su_span(chunks):
    errores = next(c for c in chunks if c.metadata["doc_id"] == "caudal-errores")
    vectorial = FakeVectorial([(chunks[0], 0.83), (errores, 0.79)])
    recuperador = RecuperadorHibrido(vectorial=vectorial, bm25=IndiceBM25(chunks), k=3)
    captura = _CapturaRetriever()

    docs = await recuperador.ainvoke("¿Qué significa el error CAU-417?", k=3, config={"callbacks": [captura]})

    assert len(docs) == 3 and vectorial.consultas == [("¿Qué significa el error CAU-417?", 6)]  # 2k candidatos
    primero = docs[0].metadata
    assert primero["doc_id"] == "caudal-errores" and primero["recuperado_por"] == ["vector", "bm25"]
    assert primero["score_vector"] == 0.79 and primero["score_bm25"] > 0
    assert captura.documentos and captura.documentos[0].metadata["score"] == primero["score"]
    with pytest.raises(NotImplementedError):
        recuperador.invoke("sync no")


async def test_modos_vector_y_bm25_reportan_el_score_de_su_via(chunks):
    vectorial = FakeVectorial([(chunks[0], 0.83)])
    servicio = ServicioRAG(RecuperadorHibrido(vectorial=vectorial, bm25=IndiceBM25(chunks), k=5))
    solo_vector = await servicio.buscar(ConsultaRAG(consulta="instalación", modo="vector", k=1))
    assert solo_vector.fragmentos[0].score == 0.83 and solo_vector.fragmentos[0].recuperado_por == ["vector"]
    solo_bm25 = await servicio.buscar(ConsultaRAG(consulta="CAU-417", modo="bm25", k=2))
    assert solo_bm25.fragmentos[0].score == solo_bm25.fragmentos[0].score_bm25
    assert vectorial.consultas == [("instalación", 1)]  # el modo bm25 no tocó Pinecone
    assert isinstance(solo_bm25, ResultadoBusqueda) and solo_bm25.latencia_ms >= 0


async def test_pinecone_busqueda_vectorial_es_async_y_normaliza_la_metadata():
    class IndiceFake:
        async def query(self, **kwargs):
            self.kwargs = kwargs
            meta = {"text": "contenido", "doc_id": "caudal-errores", "section": "CAU-417", "page": 3.0, "chunk_index": 0.0}
            return SimpleNamespace(matches=[SimpleNamespace(id="caudal-errores-p3-c0", score=0.9, metadata=meta)])

    class EmbeddingsFake:
        async def aembed_query(self, texto: str) -> list[float]:
            return [0.1, 0.2]

    indice = IndiceFake()
    resultados = await PineconeBusquedaVectorial(indice, EmbeddingsFake(), "ns-test").buscar("CAU-417", 4)
    doc, score = resultados[0]
    assert indice.kwargs == {"top_k": 4, "vector": [0.1, 0.2], "namespace": "ns-test", "include_metadata": True}
    assert score == 0.9 and doc.page_content == "contenido" and "text" not in doc.metadata
    assert doc.metadata["page"] == 3 and isinstance(doc.metadata["page"], int)
    assert doc.metadata["chunk_id"] == "caudal-errores-p3-c0"


async def test_endpoint_search_valida_y_devuelve_fuentes_con_scores(chunks, publicador):
    servicio = ServicioRAG(RecuperadorHibrido(vectorial=FakeVectorial([(chunks[0], 0.83)]), bm25=IndiceBM25(chunks)))
    async with levantar_api(orquestador(publicador), rag=servicio) as api:
        r = await api.client.post("/search", json={"consulta": "error CAU-417", "k": 3})
        assert r.status_code == 200
        cuerpo = r.json()
        assert cuerpo["modo"] == "hibrido" and len(cuerpo["fragmentos"]) == 3
        assert {"doc_id", "seccion", "chunk_id", "score", "score_vector", "score_bm25", "recuperado_por", "contenido"} <= set(
            cuerpo["fragmentos"][0]
        )
        for invalido in (
            {"consulta": "ab"},
            {"consulta": "error CAU-417", "k": 50},
            {"consulta": "error", "modo": "magia"},
            {"q": "x"},
        ):
            assert (await api.client.post("/search", json=invalido)).status_code == 422

    roto = ServicioRAG(RecuperadorHibrido(vectorial=FakeVectorial([], falla=ConnectionError("sin red")), bm25=IndiceBM25(chunks)))
    async with levantar_api(orquestador(publicador), rag=roto) as api:
        r = await api.client.post("/search", json={"consulta": "error CAU-417"})
        assert r.status_code == 503 and "ConnectionError" in r.json()["detail"]


async def test_la_respuesta_de_un_job_trae_las_fuentes_del_rag(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job = await api.esperar(await api.crear(PREGUNTA), "DONE", "FAILED")
    fuentes = job["resultado"]["fuentes"]
    assert [f["chunk_id"] for f in fuentes] == ["caudal-instalacion-p1-c0", "caudal-rendimiento-p1-c1"]
    assert fuentes[0]["doc_id"] == "caudal-instalacion" and fuentes[0]["seccion"] == "Configuración"
    assert fuentes[0]["score"] >= fuentes[1]["score"] and fuentes[0]["score_vector"] == 0.8


def _settings(**extra: Any) -> Settings:
    base = {
        "openai_api_key": "k",
        "openai_model": "m",
        "embedding_model": "e",
        "pinecone_api_key": "p",
        "index_name": "indice-test",
        "pinecone_namespace": "ns",
    }
    return Settings(_env_file=None, **{**base, **extra})


def test_armar_registros_valida_dimension_y_guarda_el_texto(chunks):
    registros = armar_registros(chunks[:2], [[0.0] * 1536, [0.1] * 1536], 1536)
    assert registros[0]["id"] == chunks[0].metadata["chunk_id"] and registros[0]["metadata"]["text"] == chunks[0].page_content
    with pytest.raises(ValueError, match="dimensiones"):
        armar_registros(chunks[:1], [[0.0] * 3], 1536)


async def test_asegurar_indice_es_idempotente_y_detecta_mismatch():
    class PineconeFake:
        def __init__(self, dimension: int) -> None:
            self.dimension, self.creados = dimension, []

        async def has_index(self, nombre):
            return True

        async def describe_index(self, nombre):
            return SimpleNamespace(dimension=self.dimension, metric="cosine")

    assert await asegurar_indice(PineconeFake(1536), _settings()) is False
    with pytest.raises(IndiceIncompatible, match="dimension=768"):
        await asegurar_indice(PineconeFake(768), _settings())


class _IndiceFake:
    """Índice async de Pinecone en memoria: upsert, stats y context manager."""

    def __init__(self, nube: "_PineconeNubeFake") -> None:
        self.nube = nube

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def upsert(self, vectors, namespace, **kwargs):
        self.nube.vectores.setdefault(namespace, {}).update({v["id"]: v for v in vectors})

    async def describe_index_stats(self):
        return SimpleNamespace(namespaces={ns: SimpleNamespace(vector_count=len(v)) for ns, v in self.nube.vectores.items()})


class _PineconeNubeFake:
    def __init__(self, existe: bool, vectores: int = 0) -> None:
        self.existe, self.creaciones = existe, 0
        self.vectores: dict[str, dict] = {"ns": {f"v{i}": {} for i in range(vectores)}} if vectores else {}

    async def has_index(self, nombre):
        return self.existe

    async def describe_index(self, nombre):
        return SimpleNamespace(dimension=1536, metric="cosine", status=SimpleNamespace(ready=True))

    async def create_index(self, **kwargs):
        self.existe, self.creaciones = True, self.creaciones + 1

    async def index(self, name):
        return _IndiceFake(self)


class _EmbeddingsFake:
    def __init__(self) -> None:
        self.llamadas = 0

    async def aembed_documents(self, textos):
        self.llamadas += 1
        return [[0.0] * 1536 for _ in textos]


async def test_preparar_indice_ingesta_sola_si_el_indice_no_existe(chunks):
    nube, emb = _PineconeNubeFake(existe=False), _EmbeddingsFake()
    resumen = await preparar_indice(_settings(), nube, emb)
    assert resumen is not None and resumen.indice_creado and resumen.chunks == len(chunks)
    assert nube.creaciones == 1 and len(nube.vectores["ns"]) == len(chunks)


async def test_preparar_indice_carga_un_namespace_vacio():
    nube, emb = _PineconeNubeFake(existe=True), _EmbeddingsFake()
    resumen = await preparar_indice(_settings(), nube, emb)
    assert resumen is not None and not resumen.indice_creado and emb.llamadas == 1


async def test_preparar_indice_no_toca_un_indice_ya_cargado():
    nube, emb = _PineconeNubeFake(existe=True, vectores=21), _EmbeddingsFake()
    assert await preparar_indice(_settings(), nube, emb) is None
    assert emb.llamadas == 0 and nube.creaciones == 0


async def test_preparar_indice_completa_una_ingesta_que_quedo_a_medias(chunks):
    nube, emb = _PineconeNubeFake(existe=True, vectores=5), _EmbeddingsFake()  # 5 de 21
    resumen = await preparar_indice(_settings(), nube, emb)
    assert resumen is not None and emb.llamadas == 1 and len(nube.vectores["ns"]) >= len(chunks)


async def test_preparar_indice_tolera_que_otro_proceso_lo_cree_a_la_vez(chunks):
    """API y worker aparte arrancando juntos en una cuenta nueva: el segundo create_index da 409."""

    class NubeConCarrera(_PineconeNubeFake):
        async def create_index(self, **kwargs):
            self.existe = True  # lo creó "el otro proceso"
            raise RuntimeError("(409) Conflict: ALREADY_EXISTS: Resource already exists")

    nube = NubeConCarrera(existe=False)
    resumen = await preparar_indice(_settings(), nube, _EmbeddingsFake())
    assert resumen is not None and resumen.chunks == len(chunks)
