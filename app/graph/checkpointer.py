"""Checkpointer de LangGraph en Redis: las conversaciones no se pierden al reiniciar el servidor.

`AsyncRedisSaver` (paquete `langgraph-checkpoint-redis`) guarda cada super-step del grafo en
Redis (módulos JSON + Search, que trae la imagen `redis:8`). Con Redis corriendo con AOF y un
volumen, el estado sobrevive a reiniciar la API, el worker y el propio contenedor de Redis:

    checkpoint:{thread_id}:...        el estado del grafo en cada paso
    checkpoint_write:{thread_id}:...  escrituras pendientes de cada nodo (tolerancia a fallos)

`asetup()` crea los índices de RediSearch que usa el saver para buscar checkpoints por hilo.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver


@asynccontextmanager
async def redis_checkpointer(redis_url: str) -> AsyncIterator[BaseCheckpointSaver]:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        yield saver
