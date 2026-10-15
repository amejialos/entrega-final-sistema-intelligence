"""Logging del proceso: un formato único para la API, el worker y los scripts."""

import logging
import warnings


def configurar_logging(nivel: str = "INFO") -> None:
    logging.basicConfig(level=nivel, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    # Ruido de librerías: cada pedido HTTP de httpx/OpenAI en INFO tapa los logs del sistema.
    for ruidoso in ("httpx", "httpx2", "httpcore", "openai", "pinecone"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    # Aviso conocido de langchain-openai al serializar la salida estructurada del supervisor.
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
