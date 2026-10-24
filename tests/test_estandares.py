"""Los "errores a evitar" de la consigna, controlados sobre TODO el código de `app/`:

- nada de llamadas síncronas que bloqueen el event loop (`requests`, `time.sleep`);
- ningún nombre de modelo ni de índice escrito en el código: todo sale de variables de entorno.
"""

import ast
import re
from pathlib import Path

import pytest

ARCHIVOS = sorted(Path(__file__).parents[1].joinpath("app").rglob("*.py"))
# Un literal que ES un id de modelo o de índice (no un texto que lo menciona en una oración).
ID_PROHIBIDO = re.compile(r"(gemini|gpt|claude|text-embedding|o\d)[-\w.]*|pre-entrega-\d+|caudal-docs", re.IGNORECASE)


def _literales(arbol: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(arbol) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_hay_codigo_para_revisar():
    assert len(ARCHIVOS) > 30


@pytest.mark.parametrize("archivo", ARCHIVOS, ids=lambda p: str(p.relative_to(p.parents[2])))
def test_sin_llamadas_bloqueantes(archivo):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            assert all(a.name != "requests" for a in nodo.names), "usá httpx (async), no requests"
        if isinstance(nodo, ast.ImportFrom):
            assert nodo.module != "requests", "usá httpx (async), no requests"
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute) and nodo.func.attr == "sleep":
            assert getattr(nodo.func.value, "id", None) != "time", "usá await asyncio.sleep, no time.sleep"


@pytest.mark.parametrize("archivo", ARCHIVOS, ids=lambda p: str(p.relative_to(p.parents[2])))
def test_sin_modelos_ni_indices_hardcodeados(archivo):
    prohibidos = [s for s in _literales(ast.parse(archivo.read_text(encoding="utf-8"))) if ID_PROHIBIDO.fullmatch(s.strip())]
    assert prohibidos == [], f"van en el .env, no en el código: {prohibidos}"
