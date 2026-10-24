"""El orquestador con modelos y RAG fake: rutas, tope de pasos, refinamiento, trazabilidad de
contribuciones, aislamiento de contexto, fuera de dominio y el diagrama del grafo real."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.researcher import crear_herramienta_busqueda
from app.agents.synthesizer import RESPUESTA_FUERA_DE_DOMINIO
from app.graph.builder import RECURSION_LIMIT, build_graph, run_config
from app.graph.state import entrada_turno
from tests.conftest import SIN_REINTENTOS, PublicadorFake
from tests.fakes import FakeBuscador, FakeSupervisorModel, FakeToolModel, decision, llamada

PREGUNTA = (
    "MARCADOR-PREGUNTA: según la documentación de Caudal, ¿cuál es el tamaño de lote por defecto? "
    "¿Cuántos lotes son 2 millones de eventos?"
)


async def ejecutar(graph, pregunta: str) -> tuple[dict[str, Any], list[str]]:
    """Corre el grafo y devuelve (estado final, nodos en orden)."""
    nodos: list[str] = []
    final: dict[str, Any] = {}
    async for modo, data in graph.astream(entrada_turno(pregunta), run_config(), stream_mode=["updates", "values"]):
        if modo == "values":
            final = data
        else:
            nodos.extend(data)
    return final, nodos


def _investigador(rondas: int = 1) -> FakeToolModel:
    ronda = [
        llamada("buscar_documentacion", {"consulta": "tamano_lote por defecto"}),
        AIMessage("tamano_lote por defecto: 500 eventos [caudal-instalacion › Configuración]."),
    ]
    return FakeToolModel(respuestas=ronda * rondas)


def _analista(rondas: int = 1) -> FakeToolModel:
    ronda = [llamada("calcular", {"expresion": "2000000 / 500"}), AIMessage("2.000.000 / 500 = 4000 lotes (calcular).")]
    return FakeToolModel(respuestas=ronda * rondas)


def _grafo(supervisor, investigador=None, analista=None, sintesis=None, buscador=None, **kwargs):
    return build_graph(
        supervisor_model=supervisor,
        research_model=investigador or _investigador(),
        analyst_model=analista or _analista(),
        synth_model=sintesis or FakeToolModel(respuestas=[AIMessage("Respuesta final: 500 por defecto, 4000 lotes.")]),
        search_tool=crear_herramienta_busqueda(buscador or FakeBuscador()),
        retry_policy=SIN_REINTENTOS,
        publicador=PublicadorFake(),
        **kwargs,
    )


async def test_ruta_supervisor_investigador_analista_cierre():
    supervisor = FakeSupervisorModel(
        decisiones=[
            decision("researcher", "Buscá el tamaño de lote por defecto (tamano_lote)."),
            decision("analyst", "Calculá cuántos lotes son 2.000.000 eventos con el lote por defecto."),
            decision("FINISH"),
        ]
    )
    sintesis = FakeToolModel(respuestas=[AIMessage("Respuesta final: 500 por defecto, 4000 lotes.")])
    final, nodos = await ejecutar(_grafo(supervisor, sintesis=sintesis), PREGUNTA)

    assert nodos == [
        "supervisor",
        "researcher",
        "supervisor",
        "analyst",
        "supervisor",
        "synthesizer",
        "evaluar_criticidad",
        "cerrar_turno",
    ]
    assert final["motivos_criticidad"] == []  # no es crítica: no pasa por aprobación
    assert final["task_completed"] is True and final["cierre_forzado"] is False
    assert final["final_answer"] == "Respuesta final: 500 por defecto, 4000 lotes."
    assert [d["next"] for d in final["decisiones"]] == ["researcher", "analyst", "FINISH"]
    assert final["historial"][-1]["respuesta"] == final["final_answer"]
    # La síntesis recibió las contribuciones con autor y fuentes, no los mensajes internos.
    entrada_sintesis = sintesis.recibidos[0][-1].content
    assert "researcher" in entrada_sintesis and "caudal-instalacion › Configuración" in entrada_sintesis
    assert "4000 lotes" in entrada_sintesis


async def test_contribuciones_se_acumulan_con_autor_tarea_y_fuentes_del_rag():
    supervisor = FakeSupervisorModel(
        decisiones=[decision("researcher", "Buscá tamano_lote."), decision("analyst", "Calculá 2000000/500."), decision("FINISH")]
    )
    final, _ = await ejecutar(_grafo(supervisor), PREGUNTA)

    investigador, analista = final["contributions"]
    assert investigador["agente"] == "researcher" and investigador["paso"] == 1
    assert investigador["tarea"] == "Buscá tamano_lote."
    assert investigador["herramientas"] == ["buscar_documentacion"]
    assert investigador["fuentes"] == ["caudal-instalacion › Configuración", "caudal-rendimiento › Backpressure y tamaño de lote"]
    # Las fuentes del RAG (con chunk_id y scores) salieron del artifact tipado de la herramienta.
    assert [f["chunk_id"] for f in investigador["fuentes_rag"]] == ["caudal-instalacion-p1-c0", "caudal-rendimiento-p1-c1"]
    assert investigador["fuentes_rag"][0]["score"] > investigador["fuentes_rag"][1]["score"]
    assert analista["agente"] == "analyst" and analista["herramientas"] == ["calcular"] and analista["estado"] == "ok"
    assert analista["fuentes_rag"] == []
    # En el historial del orquestador cada aporte queda firmado con `name`.
    nombres = [m.name for m in final["messages"] if isinstance(m, AIMessage)]
    assert nombres == ["supervisor", "researcher", "supervisor", "analyst", "supervisor", "synthesizer"]


async def test_especialistas_no_reciben_el_historial_completo():
    supervisor = FakeSupervisorModel(
        decisiones=[decision("researcher", "Buscá tamano_lote."), decision("analyst", "Calculá 2000000/500."), decision("FINISH")]
    )
    investigador, analista = _investigador(), _analista()
    await ejecutar(_grafo(supervisor, investigador=investigador, analista=analista), PREGUNTA)

    # Primera llamada de cada especialista: system prompt + UN mensaje humano con la tarea.
    primera_inv = investigador.recibidos[0]
    assert [type(m).__name__ for m in primera_inv] == ["SystemMessage", "HumanMessage"]
    assert primera_inv[1].content == "Tarea: Buscá tamano_lote."

    primera_ana = analista.recibidos[0]
    assert [type(m).__name__ for m in primera_ana] == ["SystemMessage", "HumanMessage"]
    pedido = primera_ana[1].content
    assert pedido.startswith("Tarea: Calculá 2000000/500.")
    assert "500 eventos [caudal-instalacion › Configuración]" in pedido  # el contexto necesario sí llega
    for llamada_modelo in investigador.recibidos + analista.recibidos:
        texto = " ".join(str(m.content) for m in llamada_modelo)
        assert "MARCADOR-PREGUNTA" not in texto  # no ven la conversación del usuario
        assert "[supervisor]" not in texto  # ni las decisiones del supervisor
    # El analista no ve los tool calls / ToolMessages del investigador.
    assert not any(
        isinstance(m, ToolMessage) and m.name == "buscar_documentacion"
        for llamada_modelo in analista.recibidos
        for m in llamada_modelo
    )


async def test_rubrica_pide_refinamiento_y_despues_termina():
    supervisor = FakeSupervisorModel(
        decisiones=[
            decision("researcher", "Buscá tamano_lote."),
            decision("analyst", "Calculá 2000000/500."),
            decision(
                "analyst",
                "Refiná: faltó calcular con el lote de ingesta masiva (5000): 2000000/5000.",
                razon="calculos incompletos",
            ),
            decision("FINISH"),
        ]
    )
    final, nodos = await ejecutar(_grafo(supervisor, analista=_analista(rondas=2)), PREGUNTA)

    assert nodos.count("analyst") == 2
    assert [c["agente"] for c in final["contributions"]] == ["researcher", "analyst", "analyst"]
    assert final["contributions"][2]["tarea"].startswith("Refiná")
    assert final["decisiones"][2]["rubrica"]["calculos"] is False
    # En el paso de refinamiento el supervisor vio las contribuciones previas con su autor.
    vio = supervisor.recibidos[2][-1].content
    assert "#1 researcher" in vio and "#2 analyst" in vio


async def test_tope_de_pasos_corta_el_bucle_del_supervisor():
    # Un supervisor que nunca da FINISH: sin el tope, el bucle sería infinito.
    supervisor = FakeSupervisorModel(decisiones=[decision("researcher", "Buscá de nuevo.")], repetir_ultima=True)
    sintesis = FakeToolModel(respuestas=[AIMessage("Respuesta parcial.")])
    final, nodos = await ejecutar(
        _grafo(supervisor, investigador=_investigador(rondas=10), sintesis=sintesis, max_steps=3), PREGUNTA
    )

    assert len(supervisor.recibidos) == 3  # el 4.º paso no llama al LLM
    assert nodos.count("researcher") == 3
    assert nodos[-3:] == ["synthesizer", "evaluar_criticidad", "cerrar_turno"]
    assert final["cierre_forzado"] is True and final["task_completed"] is True
    assert "cierre fue forzado" in sintesis.recibidos[0][-1].content


async def test_error_de_busqueda_queda_como_contribucion_con_error():
    supervisor = FakeSupervisorModel(decisiones=[decision("researcher", "Buscá tamano_lote."), decision("FINISH")])
    investigador = FakeToolModel(
        respuestas=[
            llamada("buscar_documentacion", {"consulta": "tamano_lote"}),
            AIMessage("La documentación no está disponible."),
        ]
    )
    graph = _grafo(supervisor, investigador=investigador, buscador=FakeBuscador(falla=ConnectionError("timeout")))
    final, _ = await ejecutar(graph, PREGUNTA)

    contribucion = final["contributions"][0]
    assert contribucion["estado"] == "error" and contribucion["fuentes"] == [] and contribucion["fuentes_rag"] == []
    assert "estado: error" in supervisor.recibidos[1][-1].content  # el supervisor lo ve


async def test_pregunta_fuera_de_dominio_no_busca_ni_llama_a_la_sintesis():
    supervisor = FakeSupervisorModel(decisiones=[decision("researcher", "Buscá el mundial", fuera_de_dominio=True)])
    sintesis = FakeToolModel(respuestas=[])  # si se la llama, falla
    buscador = FakeBuscador()
    final, nodos = await ejecutar(_grafo(supervisor, sintesis=sintesis, buscador=buscador), "¿Quién ganó el mundial de 2022?")

    assert nodos == ["supervisor", "synthesizer", "evaluar_criticidad", "cerrar_turno"]
    assert final["fuera_de_dominio"] is True and final["final_answer"] == RESPUESTA_FUERA_DE_DOMINIO
    assert final["decisiones"][0]["next"] == "FINISH"  # el validador de Pydantic corrigió el destino
    assert buscador.consultas == [] and sintesis.recibidos == []


async def test_diagrama_mermaid_del_grafo_real_y_recursion_limit():
    graph = _grafo(FakeSupervisorModel(decisiones=[decision("FINISH")]))
    assert run_config()["recursion_limit"] == RECURSION_LIMIT
    mermaid = graph.get_graph().draw_mermaid()
    for nodo in (
        "supervisor",
        "researcher",
        "analyst",
        "synthesizer",
        "evaluar_criticidad",
        "aprobacion_humana",
        "publicar_reporte",
        "cerrar_turno",
    ):
        assert nodo in mermaid
    final = await graph.ainvoke(entrada_turno("hola, ¿qué es Caudal?"), run_config())
    assert final["contributions"] == [] and final["task_completed"] is True
    assert isinstance(final["messages"][0], HumanMessage)


def test_run_config_lleva_thread_job_y_fase_a_las_trazas():
    config = run_config("hilo-1234", "a" * 32, fase="ejecutar")
    assert config["configurable"] == {"thread_id": "hilo-1234"}
    assert config["metadata"] == {"thread_id": "hilo-1234", "job_id": "a" * 32, "fase": "ejecutar"}
    assert config["run_name"] == "orquestador job=aaaaaaaa (ejecutar)"


async def test_supervisor_usa_salida_estructurada_con_el_metodo_configurado():
    supervisor = FakeSupervisorModel(decisiones=[decision("FINISH")])
    _grafo(supervisor, structured_method="function_calling")
    assert supervisor.metodo == "function_calling"
