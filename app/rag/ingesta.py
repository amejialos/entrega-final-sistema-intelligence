"""Ingesta del corpus en Pinecone (async), idempotente. Viene de la Pre-entrega 4.

- `asegurar_indice`: crea el índice serverless solo si no existe; si existe, verifica que la
  dimensión y la métrica coincidan con las del modelo de embeddings (el error más común con
  Pinecone es el mismatch de dimensiones, que recién aparece en el primer upsert).
- `ingestar`: chunks -> embeddings (`aembed_documents`) -> `upsert` con los `chunk_id`
  deterministas como id. Volver a correrla sobreescribe los mismos vectores en vez de
  duplicarlos. Todo va al namespace de `PINECONE_NAMESPACE`, nunca al namespace por defecto.

La usan `scripts/init_index.py` y `scripts/ingest.py`.
"""

import asyncio
import logging
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from app.core.config import Settings
from app.rag.corpus import load_chunks

logger = logging.getLogger(__name__)

METRICA = "cosine"
CAMPOS_METADATA = ("doc_id", "chunk_id", "source", "page", "section", "category", "title", "chunk_index")


class IndiceIncompatible(RuntimeError):
    """El índice existe pero con otra dimensión o métrica."""


class ResumenIngesta(BaseModel):
    indice: str
    namespace: str
    indice_creado: bool
    chunks: int
    documentos: list[str]
    vectores_visibles: int
    segundos: float


async def asegurar_indice(pc: Any, settings: Settings, espera_max_s: float = 120.0) -> bool:
    """Crea el índice si no existe. Devuelve True si lo creó."""
    from pinecone import ServerlessSpec

    nombre = settings.index_name
    if await pc.has_index(nombre):
        desc = await pc.describe_index(nombre)
        if desc.dimension != settings.embedding_dimension or str(desc.metric) != METRICA:
            raise IndiceIncompatible(
                f"El índice {nombre!r} existe con dimension={desc.dimension} y metric={desc.metric}, pero el "
                f"sistema espera dimension={settings.embedding_dimension} y metric={METRICA}. Usá otro INDEX_NAME."
            )
        logger.info("El índice %r ya existe (dimension=%s, metric=%s): no se recrea.", nombre, desc.dimension, desc.metric)
        return False

    logger.info("Creando el índice serverless %r...", nombre)
    await pc.create_index(
        name=nombre,
        dimension=settings.embedding_dimension,
        metric=METRICA,
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    limite = time.monotonic() + espera_max_s
    while not (await pc.describe_index(nombre)).status.ready:
        if time.monotonic() > limite:
            raise TimeoutError(f"El índice {nombre!r} no quedó listo en {espera_max_s} s.")
        await asyncio.sleep(2)
    return True


def armar_registros(chunks: list[Document], vectores: list[list[float]], dimension: int) -> list[dict[str, Any]]:
    """Registros de upsert: id, vector y metadata (incluido el texto, que Pinecone no guarda aparte)."""
    if len(chunks) != len(vectores):
        raise ValueError(f"{len(chunks)} chunks pero {len(vectores)} vectores.")
    registros = []
    for chunk, vector in zip(chunks, vectores, strict=True):
        if len(vector) != dimension:
            raise ValueError(
                f"El embedding de {chunk.metadata['chunk_id']} tiene {len(vector)} dimensiones; el índice espera {dimension}."
            )
        metadata = {campo: chunk.metadata[campo] for campo in CAMPOS_METADATA}
        metadata["text"] = chunk.page_content
        registros.append({"id": chunk.metadata["chunk_id"], "values": vector, "metadata": metadata})
    return registros


async def contar_vectores(index: Any, namespace: str) -> int:
    stats = await index.describe_index_stats()
    info = stats.namespaces.get(namespace)
    return 0 if info is None else int(info.vector_count)


async def ingestar(
    settings: Settings, pc: Any, embeddings: Embeddings, reset: bool = False, espera_max_s: float = 60.0
) -> ResumenIngesta:
    inicio = time.monotonic()
    creado = await asegurar_indice(pc, settings)
    chunks = await asyncio.to_thread(load_chunks)  # lectura de archivos + tiktoken
    index = await pc.index(name=settings.index_name)
    async with index:
        if reset:
            try:
                await index.delete(delete_all=True, namespace=settings.pinecone_namespace)
            except Exception as exc:  # 404 si el namespace todavía no existe
                if "404" not in str(exc) and "not found" not in str(exc).lower():
                    raise
        vectores = await embeddings.aembed_documents([c.page_content for c in chunks])
        registros = armar_registros(chunks, vectores, settings.embedding_dimension)
        await index.upsert(vectors=registros, namespace=settings.pinecone_namespace, batch_size=100, show_progress=False)
        # Pinecone es eventualmente consistente: se espera a que los vectores sean visibles.
        limite = time.monotonic() + espera_max_s
        while (visibles := await contar_vectores(index, settings.pinecone_namespace)) < len(registros):
            if time.monotonic() > limite:
                break
            await asyncio.sleep(2)
    return ResumenIngesta(
        indice=settings.index_name,
        namespace=settings.pinecone_namespace,
        indice_creado=creado,
        chunks=len(chunks),
        documentos=sorted({c.metadata["doc_id"] for c in chunks}),
        vectores_visibles=visibles,
        segundos=round(time.monotonic() - inicio, 1),
    )
