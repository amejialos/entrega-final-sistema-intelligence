"""Streaming del progreso: el worker registra cada nodo del grafo que termina y
`GET /tasks/{id}/stream` lo transmite por Server-Sent Events junto con los cambios de estado."""

import json
from typing import Any

from tests.conftest import PREGUNTA, RESPUESTA, levantar_api, orquestador


async def _leer_sse(api, job_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Lee el stream entero (se cierra solo al llegar a DONE/FAILED o a WAITING_APPROVAL)."""
    eventos, tipo = [], None
    async with api.client.stream("GET", f"/tasks/{job_id}/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        async for linea in r.aiter_lines():
            if linea.startswith("event: "):
                tipo = linea.removeprefix("event: ")
            elif linea.startswith("data: "):
                eventos.append((tipo, json.loads(linea.removeprefix("data: "))))
    return eventos


async def test_el_job_registra_los_pasos_del_grafo_con_la_decision_del_supervisor(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job = await api.esperar(await api.crear(PREGUNTA), "DONE", "FAILED")
        assert job["status"] == "DONE"
        nodos = [p["nodo"] for p in job["pasos"]]
        assert nodos == ["supervisor", "researcher", "supervisor", "synthesizer", "evaluar_criticidad", "cerrar_turno"]
        assert job["pasos"][0]["detalle"].startswith("-> researcher")


async def test_stream_transmite_estados_y_nodos_en_orden_y_cierra_al_terminar(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job_id = await api.crear(PREGUNTA)
        eventos = await _leer_sse(api, job_id)

    tipos = [t for t, _ in eventos]
    assert tipos[0] == "estado" and eventos[0][1]["status"] == "PENDING"
    assert [d["nodo"] for t, d in eventos if t == "nodo"][-1] == "cerrar_turno"
    assert [d["status"] for t, d in eventos if t == "estado"] == ["PENDING", "RUNNING", "DONE"]
    assert tipos[-1] == "fin" and eventos[-1][1]["status"] == "DONE"
    assert eventos[-1][1]["resultado"]["respuesta"]
    momentos = [d["t"] for t, d in eventos if t in ("estado", "nodo")]
    assert momentos == sorted(momentos)  # estados y nodos intercalados por hora


async def test_stream_se_cierra_en_la_pausa_de_aprobacion(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job_id = await api.crear(PREGUNTA, publicar=True)
        eventos = await _leer_sse(api, job_id)
    assert eventos[-1][0] == "fin" and eventos[-1][1]["status"] == "WAITING_APPROVAL"
    assert eventos[-1][1]["aprobacion_pendiente"]["accion"] == "publicar_reporte"


async def test_stream_de_un_job_inexistente_es_404(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        r = await api.client.get(f"/tasks/{'0' * 32}/stream")
        assert r.status_code == 404


async def test_stream_transmite_los_tokens_de_la_sintesis_antes_de_que_termine(publicador):
    """Lo que escribe el modelo de síntesis llega como eventos `token`; juntos forman la respuesta final."""
    async with levantar_api(orquestador(publicador)) as api:
        job_id = await api.crear(PREGUNTA)
        eventos = await _leer_sse(api, job_id)
        job = (await api.client.get(f"/tasks/{job_id}")).json()

    texto = "".join(d["texto"] for t, d in eventos if t == "token")
    # Exactamente la respuesta: ni el supervisor ni el investigador (que también llaman a un modelo) se cuelan.
    assert texto == job["resultado"]["respuesta"] == RESPUESTA
    tipos = [t if t != "nodo" else f"nodo:{d['nodo']}" for t, d in eventos]
    assert tipos.index("token") < tipos.index("nodo:synthesizer")  # se ve mientras se escribe


class _PedidoConectado:
    async def is_disconnected(self) -> bool:
        return False


async def test_un_reintento_que_escribe_mas_texto_igual_reinicia_el_borrador():
    """El largo no alcanza para detectar un reintento: el borrador lleva un número de generación."""
    import fakeredis

    from app.api.routes_tasks import _eventos_sse
    from app.worker.jobs import JobStore
    from app.worker.models import Estado

    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = JobStore(redis)
    job = await store.crear(PREGUNTA)
    await store.agregar_borrador(job.job_id, "Prim")
    eventos = _eventos_sse(job.job_id, store, _PedidoConectado())
    primeros = [await anext(eventos), await anext(eventos)]
    assert primeros[0].startswith("event: estado") and '"Prim"' in primeros[1]  # estado PENDING primero

    await store.agregar_borrador(job.job_id, "Un texto más largo que el primero", reiniciar=True)
    await store.actualizar(job.job_id, Estado.DONE)
    resto = [e async for e in eventos]
    i = next(n for n, e in enumerate(resto) if e.startswith("event: reinicio"))
    assert '"Un texto más largo que el primero"' in resto[i + 1]  # el texto nuevo, entero, después del reinicio
    assert resto[-1].startswith("event: fin")
    await redis.aclose()


async def test_el_stream_documenta_todos_sus_eventos_en_openapi(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        esquema = (await api.client.get("/openapi.json")).json()
    descripcion = esquema["paths"]["/tasks/{job_id}/stream"]["get"]["responses"]["200"]["description"]
    for evento in ("estado", "nodo", "token", "reinicio", "fin"):
        assert f"`{evento}`" in descripcion
