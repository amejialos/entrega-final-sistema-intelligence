"""Human-in-the-loop: la tarea crítica pausa en WAITING_APPROVAL hasta la señal externa; aprobar
reanuda desde el checkpoint y publica; rechazar termina explicándolo; una no crítica no pausa."""

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.hitl import motivos_criticidad
from app.worker.jobs import JobStore
from app.worker.worker import Worker
from tests.conftest import PRECIOS_TEST, PREGUNTA, RESPUESTA, levantar_api, orquestador

# --- El criterio de tarea crítica ---


@pytest.mark.parametrize(
    ("pregunta", "publicar", "critica"),
    [
        (PREGUNTA, False, False),
        (PREGUNTA, True, True),  # pedido explícito en el body
        ("Calculá los lotes y publicá el reporte para el equipo.", False, True),
        ("Respondé y envialo por mail al equipo de datos.", False, True),
        ("Notificar al canal cuando esté el cálculo de lotes.", False, True),
        ("Exportá la estimación de tiempos de procesamiento.", False, True),
        ("¿Qué dice la documentación sobre la publicación de eventos?", False, False),  # sustantivo, no pedido
        ("¿Cómo es el envío de eventos en lote?", False, False),
    ],
)
def test_criterio_de_tarea_critica(pregunta, publicar, critica):
    motivos = motivos_criticidad(pregunta, publicar)
    assert bool(motivos) is critica
    if critica:
        assert all("publicar_reporte" in m for m in motivos)


# --- El flujo completo por la API ---


async def test_tarea_no_critica_termina_sin_pausa(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job_id = await api.crear(PREGUNTA)
        job = await api.esperar(job_id, "DONE", "WAITING_APPROVAL", "FAILED")

        assert job["status"] == "DONE"
        assert "WAITING_APPROVAL" not in [e["status"] for e in job["eventos"]]
        assert job["resultado"]["respuesta"] == RESPUESTA
        assert job["resultado"]["reporte_publicado"] is None
        assert job["metricas"]["espera_aprobacion_s"] is None
        assert publicador.publicados == []


async def test_tarea_critica_pausa_hasta_la_aprobacion_y_despues_publica(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job_id = await api.crear(PREGUNTA, publicar=True)
        job = await api.esperar(job_id, "WAITING_APPROVAL", "DONE", "FAILED")

        # Pausa: nada se publicó y el revisor ve qué se va a hacer y por qué.
        assert job["status"] == "WAITING_APPROVAL"
        pendiente = job["aprobacion_pendiente"]
        assert pendiente["accion"] == "publicar_reporte"
        assert pendiente["borrador"] == RESPUESTA
        assert any("publicar=true" in m for m in pendiente["motivos"])
        assert job["metricas"]["corridas"] == 1 and job["metricas"]["llamadas_llm"] == 5
        assert publicador.publicados == []
        # Sigue pausado: sin la señal externa no avanza solo, y la conversación queda ocupada.
        await asyncio.sleep(0.3)
        assert (await api.client.get(f"/tasks/{job_id}")).json()["status"] == "WAITING_APPROVAL"
        assert await api.store.job_activo(job["thread_id"]) == job_id

        r = await api.client.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "ok", "revisor": "ana"})
        assert r.status_code == 202 and r.json()["status"] == "PENDING"

        job = await api.esperar(job_id, "DONE", "FAILED")
        assert job["status"] == "DONE"
        assert job["aprobacion_pendiente"] is None
        assert job["resultado"]["reporte_publicado"] == f"fake:reporte:{job_id}"
        assert job["resultado"]["aprobacion"]["aprobado"] is True
        assert job["resultado"]["aprobacion"]["revisor"] == "ana"
        assert [e["status"] for e in job["eventos"]] == ["PENDING", "RUNNING", "WAITING_APPROVAL", "PENDING", "RUNNING", "DONE"]
        # Las métricas suman la corrida inicial y la reanudación (que no llamó a ningún LLM).
        assert job["metricas"]["corridas"] == 2 and job["metricas"]["llamadas_llm"] == 5
        assert job["metricas"]["espera_aprobacion_s"] >= 0.3
        # Se publicó una sola vez, con la respuesta y sus fuentes.
        assert len(publicador.publicados) == 1
        publicado_id, reporte = publicador.publicados[0]
        assert publicado_id == job_id and reporte["respuesta"] == RESPUESTA
        assert "caudal-instalacion › Configuración" in reporte["fuentes"]


async def test_rechazo_termina_sin_publicar_y_lo_explica(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        job_id = await api.crear("Buscá el tamano_lote por defecto y publicá el reporte.")
        await api.esperar(job_id, "WAITING_APPROVAL")

        r = await api.client.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "faltan fuentes"})
        assert r.status_code == 202

        job = await api.esperar(job_id, "DONE", "FAILED")
        assert job["status"] == "DONE"
        respuesta = job["resultado"]["respuesta"]
        assert respuesta.startswith("La acción 'publicar_reporte' fue RECHAZADA por revisión humana (comentario: faltan fuentes)")
        assert RESPUESTA in respuesta  # el borrador queda, marcado como tal
        assert job["resultado"]["reporte_publicado"] is None
        assert job["resultado"]["aprobacion"]["aprobado"] is False
        assert publicador.publicados == []


async def test_aprobar_un_job_que_no_espera_aprobacion_responde_409(publicador):
    async with levantar_api(orquestador(publicador)) as api:
        normal = await api.crear(PREGUNTA)
        await api.esperar(normal, "DONE")
        r = await api.client.post(f"/tasks/{normal}/approve", json={"aprobado": True})
        assert r.status_code == 409

    # Otra API (los modelos fake están guionados para una sola corrida por grafo).
    async with levantar_api(orquestador(publicador)) as api:
        critico = await api.crear(PREGUNTA, publicar=True)
        await api.esperar(critico, "WAITING_APPROVAL")
        # Dos aprobaciones simultáneas: solo una reanuda el hilo.
        respuestas = await asyncio.gather(
            api.client.post(f"/tasks/{critico}/approve", json={"aprobado": True}),
            api.client.post(f"/tasks/{critico}/approve", json={"aprobado": True}),
        )
        assert sorted(r.status_code for r in respuestas) == [202, 409]
        await api.esperar(critico, "DONE")
        assert len(publicador.publicados) == 1


@pytest.mark.parametrize(
    "body",
    [{}, {"aprobado": "tal vez"}, {"aprobado": 1}, {"aprobado": True, "revisor": ""}, {"aprobado": True, "firma": "x"}],
)
async def test_body_de_aprobacion_invalido_responde_422(publicador, body):
    async with levantar_api(orquestador(publicador)) as api:
        job_id = await api.crear(PREGUNTA, publicar=True)
        await api.esperar(job_id, "WAITING_APPROVAL")
        assert (await api.client.post(f"/tasks/{job_id}/approve", json=body)).status_code == 422
        assert (await api.client.get(f"/tasks/{job_id}")).json()["status"] == "WAITING_APPROVAL"


async def test_la_pausa_sobrevive_a_un_reinicio_porque_vive_en_el_checkpoint(publicador):
    """Otro Worker, con OTRA instancia del grafo, reanuda el hilo: el estado no estaba en memoria
    del proceso que pausó sino en el checkpointer (en producción, Redis)."""
    checkpointer = InMemorySaver()  # compartido, como el Redis real entre dos procesos
    async with levantar_api(orquestador(publicador, checkpointer)) as api:
        job_id = await api.crear(PREGUNTA, publicar=True)
        await api.esperar(job_id, "WAITING_APPROVAL")
        await api.worker.detener()  # "se apaga" el proceso que pausó

        nuevo = Worker(
            api.redis, JobStore(api.redis), orquestador(publicador, checkpointer, sin_llm=True), PRECIOS_TEST, espera_cola_s=0.05
        )
        api.app.state.recursos.worker = nuevo  # la API ahora encola para el proceso nuevo
        await nuevo.iniciar()
        try:
            r = await api.client.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
            assert r.status_code == 202
            job = await api.esperar(job_id, "DONE", "FAILED")
        finally:
            await nuevo.detener()

        assert job["status"] == "DONE"
        assert job["resultado"]["respuesta"] == RESPUESTA  # vino del checkpoint: ningún LLM se volvió a llamar
        assert job["resultado"]["reporte_publicado"] == f"fake:reporte:{job_id}"
