"""Conversaciones por `thread_id`: los seguimientos usan el historial guardado en el checkpoint, el
estado del turno se reinicia (Overwrite), la conversación sobrevive a un "reinicio" de la API y
dos turnos del mismo hilo no corren a la vez."""

import asyncio

from fakeredis import FakeAsyncRedis
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.researcher import crear_herramienta_busqueda
from app.graph.builder import build_graph
from tests.conftest import PREGUNTA, RESPUESTA, SIN_REINTENTOS, PublicadorFake, grafo_simple, levantar_api
from tests.fakes import FakeBuscador, FakeSupervisorModel, FakeToolModel, decision, llamada

SEGUIMIENTO = "¿Y cuántos lotes son 2.000.000 de eventos con ese tamaño?"
RESPUESTA_2 = "Con tamano_lote 500, 2.000.000 de eventos son 4000 lotes (calcular: 2000000 / 500)."


def grafo_dos_turnos(checkpointer=None):
    """Turno 1: supervisor -> investigador -> síntesis. Turno 2 (seguimiento): supervisor -> analista -> síntesis."""
    supervisor = FakeSupervisorModel(
        decisiones=[
            decision("researcher", "Buscá tamano_lote."),
            decision("FINISH"),
            decision("analyst", "Calculá 2000000 / 500 (tamano_lote 500 del turno anterior)."),
            decision("FINISH"),
        ]
    )
    analista = FakeToolModel(
        respuestas=[llamada("calcular", {"expresion": "2000000 / 500"}), AIMessage("4000 lotes (calcular).")]
    )
    sintesis = FakeToolModel(respuestas=[AIMessage(RESPUESTA), AIMessage(RESPUESTA_2)])
    graph = build_graph(
        supervisor_model=supervisor,
        research_model=FakeToolModel(
            respuestas=[
                llamada("buscar_documentacion", {"consulta": "tamano_lote"}),
                AIMessage("500 [caudal-instalacion › Configuración]"),
            ]
        ),
        analyst_model=analista,
        synth_model=sintesis,
        search_tool=crear_herramienta_busqueda(FakeBuscador()),
        retry_policy=SIN_REINTENTOS,
        checkpointer=checkpointer or InMemorySaver(),
        publicador=PublicadorFake(),
    )
    return graph, supervisor, analista, sintesis


async def test_el_seguimiento_en_el_mismo_thread_usa_el_historial():
    graph, supervisor, analista, sintesis = grafo_dos_turnos()
    async with levantar_api(graph) as api:
        r1 = await api.client.post("/tasks", json={"pregunta": PREGUNTA})
        thread_id = r1.json()["thread_id"]
        job1 = await api.esperar(r1.json()["job_id"], "DONE", "FAILED")
        assert job1["resultado"]["turno"] == 1

        r2 = await api.client.post("/tasks", json={"pregunta": SEGUIMIENTO, "thread_id": thread_id})
        assert r2.status_code == 202 and r2.json()["thread_id"] == thread_id
        job2 = await api.esperar(r2.json()["job_id"], "DONE", "FAILED")

        assert job2["status"] == "DONE" and job2["thread_id"] == thread_id
        assert job2["resultado"]["respuesta"] == RESPUESTA_2
        assert job2["resultado"]["turno"] == 2
        # El estado del TURNO se reinició: solo la contribución del analista, sin la del turno 1.
        assert [c["agente"] for c in job2["resultado"]["contribuciones"]] == ["analyst"]
        assert job2["resultado"]["ruta_supervisor"] == ["analyst", "FINISH"]
        # El supervisor del turno 2 vio la conversación previa (salió del checkpoint).
        visto = supervisor.recibidos[2][-1].content
        assert f"Turno 1 - Pregunta: {PREGUNTA}" in visto and RESPUESTA in visto
        assert f"Pregunta actual del usuario:\n{SEGUIMIENTO}" in visto
        # El analista recibió la respuesta previa como contexto (no hubo investigación nueva).
        assert RESPUESTA in analista.recibidos[0][-1].content
        assert RESPUESTA in sintesis.recibidos[1][-1].content

        conversacion = (await api.client.get(f"/conversations/{thread_id}")).json()
        assert [t["pregunta"] for t in conversacion["turnos"]] == [PREGUNTA, SEGUIMIENTO]
        assert [t["numero"] for t in conversacion["turnos"]] == [1, 2]
        assert conversacion["turnos"][0]["fuentes"] == [
            "caudal-instalacion › Configuración",
            "caudal-rendimiento › Backpressure y tamaño de lote",
        ]
        assert conversacion["turnos"][1]["fuentes"] == []  # el seguimiento no buscó
        assert conversacion["jobs"] == [job1["job_id"], job2["job_id"]]
        assert conversacion["job_activo"] is None and conversacion["pausado_en"] == []
        assert conversacion["checkpoints"] > 10
        # 2 preguntas + supervisor/especialista/síntesis de cada turno, firmados con `name`.
        assert conversacion["mensajes"] == 2 + 4 + 4


async def test_la_conversacion_sobrevive_a_reiniciar_la_api():
    """Dos APIs distintas (dos procesos) sobre el mismo Redis y el mismo checkpointer."""
    checkpointer, redis = InMemorySaver(), FakeAsyncRedis(decode_responses=True)
    graph1, *_ = grafo_dos_turnos(checkpointer)
    async with levantar_api(graph1, redis=redis) as api:
        r1 = await api.client.post("/tasks", json={"pregunta": PREGUNTA, "thread_id": "demo-reinicio"})
        await api.esperar(r1.json()["job_id"], "DONE")

    # "Reinicio": API, worker y grafo nuevos. Los modelos del grafo nuevo solo tienen guion para el turno 2.
    graph2, supervisor2, *_ = grafo_dos_turnos(checkpointer)
    supervisor2.decisiones = supervisor2.decisiones[2:]
    async with levantar_api(graph2, redis=redis) as api:
        antes = (await api.client.get("/conversations/demo-reinicio")).json()
        assert [t["pregunta"] for t in antes["turnos"]] == [PREGUNTA]
        r2 = await api.client.post("/tasks", json={"pregunta": SEGUIMIENTO, "thread_id": "demo-reinicio"})
        job2 = await api.esperar(r2.json()["job_id"], "DONE", "FAILED")
        assert job2["status"] == "DONE" and job2["resultado"]["turno"] == 2
        assert RESPUESTA in supervisor2.recibidos[0][-1].content  # el historial vino del checkpoint
    await redis.aclose()


async def test_dos_turnos_del_mismo_thread_no_corren_a_la_vez():
    compuerta = asyncio.Event()
    async with levantar_api(grafo_simple(compuerta)) as api:
        r1 = await api.client.post("/tasks", json={"pregunta": PREGUNTA, "thread_id": "hilo-ocupado"})
        await api.esperar(r1.json()["job_id"], "RUNNING")

        r2 = await api.client.post("/tasks", json={"pregunta": SEGUIMIENTO, "thread_id": "hilo-ocupado"})
        assert r2.status_code == 409
        assert r1.json()["job_id"] in r2.json()["detail"]
        # Otra conversación no se ve afectada.
        assert (await api.client.post("/tasks", json={"pregunta": SEGUIMIENTO})).status_code == 202

        compuerta.set()
        await api.esperar(r1.json()["job_id"], "DONE")
        r3 = await api.client.post("/tasks", json={"pregunta": SEGUIMIENTO, "thread_id": "hilo-ocupado"})
        assert r3.status_code == 202  # terminado el turno, el hilo quedó libre


async def test_un_turno_fallido_no_contamina_el_siguiente():
    """El turno 1 falla dentro del investigador (queda un checkpoint a medio camino). El turno 2
    del mismo hilo arranca de cero: la entrada nueva descarta las tareas pendientes del fallido."""
    supervisor = FakeSupervisorModel(decisiones=[decision("researcher", "Buscá tamano_lote."), decision("FINISH")])
    graph = build_graph(
        supervisor_model=supervisor,
        research_model=FakeToolModel(respuestas=[]),  # cualquier llamada falla (AssertionError)
        analyst_model=FakeToolModel(respuestas=[]),
        synth_model=FakeToolModel(respuestas=[AIMessage("respuesta del turno 2")]),
        search_tool=crear_herramienta_busqueda(FakeBuscador()),
        retry_policy=SIN_REINTENTOS,
        checkpointer=InMemorySaver(),
        publicador=PublicadorFake(),
    )
    async with levantar_api(graph) as api:
        r1 = await api.client.post("/tasks", json={"pregunta": PREGUNTA, "thread_id": "hilo-con-falla"})
        job1 = await api.esperar(r1.json()["job_id"], "FAILED", "DONE")
        assert job1["status"] == "FAILED" and "AssertionError" in job1["error"]
        assert (await api.client.get("/conversations/hilo-con-falla")).json()["pausado_en"] == ["researcher"]

        r2 = await api.client.post("/tasks", json={"pregunta": SEGUIMIENTO, "thread_id": "hilo-con-falla"})
        job2 = await api.esperar(r2.json()["job_id"], "DONE", "FAILED")
        assert job2["status"] == "DONE", job2["error"]
        assert job2["resultado"]["respuesta"] == "respuesta del turno 2"
        assert job2["resultado"]["contribuciones"] == []  # el investigador fallido no se re-ejecutó
        conversacion = (await api.client.get("/conversations/hilo-con-falla")).json()
        assert [t["pregunta"] for t in conversacion["turnos"]] == [SEGUIMIENTO]
        assert conversacion["jobs"] == [job1["job_id"], job2["job_id"]]
