"""Integración con un Redis REAL (`AsyncRedisSaver`): se saltea si no hay Redis con JSON + Search.

    docker compose up -d redis && uv run pytest -q tests/test_redis_checkpoint.py

Con OTRA instancia del grafo y OTRO checkpointer conectados al mismo Redis: el estado sale de
Redis, no de memoria. Los LLM siguen siendo fakes (sin red hacia proveedores).
"""

import os
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command
from redis.asyncio import Redis

from app.graph.builder import run_config
from app.graph.checkpointer import redis_checkpointer
from app.graph.state import entrada_turno
from tests.conftest import PREGUNTA, RESPUESTA, orquestador
from tests.fakes import decision

REDIS_URL = os.environ.get("REDIS_TEST_URL", "redis://localhost:6379/0")


async def _redis_disponible() -> bool:
    cliente = Redis.from_url(REDIS_URL, socket_connect_timeout=0.5, decode_responses=True)
    try:
        await cliente.ping()
        modulos = {m["name"] for m in await cliente.module_list()}
        return {"search", "ReJSON"} <= modulos
    except Exception:
        return False
    finally:
        await cliente.aclose()


@pytest.fixture
async def redis_real():
    if not await _redis_disponible():
        pytest.skip(f"No hay Redis con JSON + Search en {REDIS_URL}")
    cliente = Redis.from_url(REDIS_URL, decode_responses=True)
    yield cliente
    await cliente.aclose()


async def test_la_pausa_hitl_se_reanuda_desde_redis_con_otra_instancia(redis_real, publicador):
    thread_id = f"test-{uuid.uuid4().hex}"
    config = run_config(thread_id, "a" * 32)

    async with redis_checkpointer(REDIS_URL) as saver:
        grafo = orquestador(publicador, checkpointer=saver)
        await grafo.ainvoke(entrada_turno(PREGUNTA, job_id="a" * 32, publicar=True), config)
        pausa = await grafo.aget_state(config)
        assert pausa.next == ("aprobacion_humana",)
        assert pausa.tasks[0].interrupts[0].value["accion"] == "publicar_reporte"

    assert [k async for k in redis_real.scan_iter(match=f"checkpoint:{thread_id}:*")], "el checkpoint tendría que estar en Redis"

    # "Otro proceso": checkpointer y grafo nuevos (sin LLM), mismo Redis.
    async with redis_checkpointer(REDIS_URL) as saver:
        grafo = orquestador(publicador, checkpointer=saver, sin_llm=True)
        final = await grafo.ainvoke(Command(resume={"aprobado": True, "comentario": "ok", "revisor": "ana"}), config)
        assert final["final_answer"] == RESPUESTA
        assert final["reporte_publicado"] == f"fake:reporte:{'a' * 32}"
        await saver.adelete_thread(thread_id)  # no dejar basura de los tests en Redis


async def test_dos_turnos_persisten_en_redis_y_overwrite_reinicia_el_turno(redis_real, publicador):
    thread_id = f"test-{uuid.uuid4().hex}"
    async with redis_checkpointer(REDIS_URL) as saver:
        grafo = orquestador(publicador, checkpointer=saver)
        await grafo.ainvoke(entrada_turno(PREGUNTA, job_id="1" * 32), run_config(thread_id, "1" * 32))

    # Turno 2 con otro grafo y otro checkpointer: el historial sale de Redis.
    async with redis_checkpointer(REDIS_URL) as saver:
        grafo = orquestador(publicador, checkpointer=saver, decisiones=[decision("FINISH")], sintesis=[AIMessage("segunda")])
        final = await grafo.ainvoke(entrada_turno("¿Y eso cuántos lotes son?", job_id="2" * 32), run_config(thread_id, "2" * 32))
        assert [t["pregunta"] for t in final["historial"]] == [PREGUNTA, "¿Y eso cuántos lotes son?"]
        assert final["contributions"] == []  # Overwrite([]) reinició el turno, también serializado en Redis
        assert final["final_answer"] == "segunda"
        await saver.adelete_thread(thread_id)
