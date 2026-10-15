"""Exporta el diagrama Mermaid del grafo REAL (`draw_mermaid()` del grafo compilado).

    uv run python -m scripts.exportar_grafo      # escribe docs/grafo.mmd

Se arma el mismo `build_graph` que usa la API, con modelos y RAG fake (sin red ni keys): la
topología no depende de los modelos, así que el diagrama es exactamente el que corre. El README
incluye este mismo texto en un bloque ```mermaid.
"""

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.researcher import crear_herramienta_busqueda
from app.graph.builder import build_graph
from tests.fakes import FakeBuscador, FakeSupervisorModel, FakeToolModel

SALIDA = Path(__file__).resolve().parents[1] / "docs" / "grafo.mmd"


def main() -> None:
    graph = build_graph(
        supervisor_model=FakeSupervisorModel(decisiones=[]),
        research_model=FakeToolModel(respuestas=[]),
        analyst_model=FakeToolModel(respuestas=[]),
        synth_model=FakeToolModel(respuestas=[]),
        search_tool=crear_herramienta_busqueda(FakeBuscador()),
        checkpointer=InMemorySaver(),
    )
    mermaid = graph.get_graph().draw_mermaid()
    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text(mermaid, encoding="utf-8")
    print(mermaid)
    print(f"-> {SALIDA.relative_to(SALIDA.parents[1])}")


if __name__ == "__main__":
    main()
