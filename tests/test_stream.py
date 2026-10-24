"""Streaming del progreso: el worker registra cada nodo del grafo que termina y
`GET /tasks/{id}/stream` lo transmite por Server-Sent Events junto con los cambios de estado."""

import json
from typing import Any

from tests.conftest import PREGUNTA, levantar_api, orquestador


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
