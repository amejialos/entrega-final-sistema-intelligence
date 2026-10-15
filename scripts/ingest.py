"""Ingesta del corpus (`data/`) en Pinecone: crea el índice si falta y sube los chunks (idempotente).

    uv run python -m scripts.ingest            # upsert de los 21 chunks (mismos ids: no duplica)
    uv run python -m scripts.ingest --reset    # vacía el namespace antes (si cambió el chunking)
    uv run python -m scripts.ingest --solo-indice   # solo verifica/crea el índice, sin embeddings

Usa `INDEX_NAME`, `PINECONE_NAMESPACE`, `EMBEDDING_MODEL` y `EMBEDDING_DIMENSION` del `.env`.
Si el índice de la Pre-entrega 4 ya está cargado, no hace falta correrla (la API lo reutiliza).
"""

import argparse
import asyncio
import logging

from pinecone import AsyncPinecone

from app.core.config import get_settings
from app.core.logging import configurar_logging
from app.core.providers import build_embeddings
from app.rag.ingesta import asegurar_indice, contar_vectores, ingestar


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="vaciar el namespace antes de subir")
    parser.add_argument("--solo-indice", action="store_true", help="solo crear/verificar el índice")
    args = parser.parse_args()
    configurar_logging()
    settings = get_settings()

    async with AsyncPinecone(api_key=settings.pinecone_api_key.get_secret_value()) as pc:
        if args.solo_indice:
            creado = await asegurar_indice(pc, settings)
            async with await pc.index(name=settings.index_name) as index:
                vectores = await contar_vectores(index, settings.pinecone_namespace)
            print(
                f"Índice {settings.index_name}: {'creado ahora' if creado else 'ya existía'} · "
                f"namespace {settings.pinecone_namespace}: {vectores} vectores"
            )
            return
        resumen = await ingestar(settings, pc, build_embeddings(settings), reset=args.reset)
    logging.getLogger(__name__).info("Ingesta terminada")
    print(resumen.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
