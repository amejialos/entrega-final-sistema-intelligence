"""Lee de Phoenix (API GraphQL, async) costo, tokens y latencia de las trazas y arma un reporte.

    uv run python -m scripts.phoenix_report                     # proyecto PHOENIX_PROJECT_NAME
    uv run python -m scripts.phoenix_report --proyecto otro

- Por traza: nombre, job, conversación, fase, latencia, tokens y costo (lo calcula Phoenix con su
  tabla de precios por modelo).
- Por nodo del grafo: llamadas al LLM, tokens y costo (dónde se concentra el gasto).
- Capa RAG: cuántos spans RETRIEVER hubo y cuánto tardaron.
- Control cruzado: el costo y los tokens que devolvió la API en `metricas` (docs/pruebas_e2e.json)
  contra los que calculó Phoenix para el mismo job.

Salida: docs/phoenix_report.md, .json y .html (la página HTML sirve para la captura del costo
exacto: la UI de Phoenix redondea a centavos y muestra "<$0.01").
"""

import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

DOCS = Path(__file__).resolve().parents[1] / "docs"
NODOS = (
    "supervisor",
    "researcher",
    "analyst",
    "synthesizer",
    "evaluar_criticidad",
    "aprobacion_humana",
    "publicar_reporte",
    "cerrar_turno",
)


async def gql(client: httpx.AsyncClient, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    r = await client.post("/graphql", json={"query": query, "variables": variables or {}})
    r.raise_for_status()
    datos = r.json()
    if datos.get("errors"):
        raise RuntimeError(datos["errors"])
    return datos["data"]


async def proyecto(client: httpx.AsyncClient, nombre: str) -> dict[str, Any]:
    datos = await gql(
        client,
        """{ projects { edges { node { id name traceCount
              costSummary { total { cost tokens } prompt { cost tokens } completion { cost tokens } }
              p50: latencyMsQuantile(probability: 0.5)
              p95: latencyMsQuantile(probability: 0.95) } } } }""",
    )
    for edge in datos["projects"]["edges"]:
        if edge["node"]["name"] == nombre:
            return edge["node"]
    raise SystemExit(f"No existe el proyecto {nombre!r} en Phoenix")


async def spans(client: httpx.AsyncClient, project_id: str) -> list[dict[str, Any]]:
    query = """query($id: ID!, $after: String) { node(id: $id) { ... on Project {
        spans(first: 500, after: $after) { pageInfo { hasNextPage endCursor }
          edges { node { name spanKind spanId parentId latencyMs startTime statusCode
                         tokenCountPrompt tokenCountCompletion
                         costSummary { total { cost } }
                         metadata context { traceId } } } } } } }"""
    resultado, cursor = [], None
    while True:
        pagina = (await gql(client, query, {"id": project_id, "after": cursor}))["node"]["spans"]
        resultado += [e["node"] for e in pagina["edges"]]
        if not pagina["pageInfo"]["hasNextPage"]:
            return resultado
        cursor = pagina["pageInfo"]["endCursor"]


def nodo_del_grafo(span: dict[str, Any], por_id: dict[str, dict[str, Any]]) -> str:
    """Sube por los padres hasta el nodo del orquestador al que pertenece el span."""
    actual: dict[str, Any] | None = span
    while actual is not None:
        if actual["name"] in NODOS:
            return actual["name"]
        actual = por_id.get(actual["parentId"]) if actual["parentId"] else None
    return "(fuera del grafo)"


async def analizar(url: str, nombre: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        proj = await proyecto(client, nombre)
        todos = await spans(client, proj["id"])
    por_id = {s["spanId"]: s for s in todos}
    raices = sorted((s for s in todos if not s["parentId"]), key=lambda s: s["startTime"])

    costo_traza: dict[str, float] = defaultdict(float)
    tokens_traza: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    llm_traza: dict[str, int] = defaultdict(int)
    por_nodo: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    recuperaciones = []
    for s in todos:
        traza = s["context"]["traceId"]
        if s["spanKind"].upper() == "LLM":
            costo = ((s["costSummary"] or {}).get("total") or {}).get("cost") or 0.0
            costo_traza[traza] += costo
            tokens_traza[traza][0] += s["tokenCountPrompt"] or 0
            tokens_traza[traza][1] += s["tokenCountCompletion"] or 0
            llm_traza[traza] += 1
            nodo = por_nodo[nodo_del_grafo(s, por_id)]
            nodo["llamadas_llm"] += 1
            nodo["tokens_entrada"] += s["tokenCountPrompt"] or 0
            nodo["tokens_salida"] += s["tokenCountCompletion"] or 0
            nodo["costo_usd"] += costo
        if s["spanKind"].upper() == "RETRIEVER":
            recuperaciones.append(s["latencyMs"] or 0)

    trazas = []
    for r in raices:
        meta = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {})
        traza = r["context"]["traceId"]
        trazas.append(
            {
                "trace_id": traza,
                "nombre": r["name"],
                "job_id": meta.get("job_id"),
                "thread_id": meta.get("thread_id"),
                "fase": meta.get("fase"),
                "inicio": r["startTime"],
                "estado": r["statusCode"],
                "latencia_s": round((r["latencyMs"] or 0) / 1000, 2),
                "llamadas_llm": llm_traza[traza],
                "tokens_entrada": tokens_traza[traza][0],
                "tokens_salida": tokens_traza[traza][1],
                "costo_usd": round(costo_traza[traza], 6),
            }
        )
    por_job: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for t in trazas:
        if t["job_id"]:
            j = por_job[t["job_id"]]
            j["costo_usd"] += t["costo_usd"]
            j["tokens_entrada"] += t["tokens_entrada"]
            j["tokens_salida"] += t["tokens_salida"]
            j["llamadas_llm"] += t["llamadas_llm"]
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "proyecto": nombre,
        "trazas_en_proyecto": proj["traceCount"],
        "costo_total_usd": proj["costSummary"]["total"]["cost"],
        "tokens_totales": proj["costSummary"]["total"]["tokens"],
        "latencia_ms": {"p50": proj["p50"], "p95": proj["p95"]},
        "trazas": trazas,
        "por_job": {k: {m: round(v, 6) for m, v in d.items()} for k, d in por_job.items()},
        "por_nodo": {k: {m: round(v, 6) for m, v in d.items()} for k, d in por_nodo.items()},
        "rag": {
            "spans_retriever": len(recuperaciones),
            "latencia_media_ms": round(statistics.mean(recuperaciones), 1) if recuperaciones else None,
            "latencia_max_ms": round(max(recuperaciones), 1) if recuperaciones else None,
        },
    }


def control_cruzado(rep: dict[str, Any]) -> list[dict[str, Any]]:
    """Costo y tokens según la API (metricas del job) vs. según Phoenix (spans LLM de sus trazas)."""
    archivo = DOCS / "pruebas_e2e.json"
    if not archivo.exists():
        return []
    filas = []
    for r in json.loads(archivo.read_text(encoding="utf-8")):
        job = r.get("job_id")
        if not job or job not in rep["por_job"]:
            continue
        ph = rep["por_job"][job]
        filas.append(
            {
                "escenario": r["escenario"],
                "job_id": job,
                "api_costo_usd": r.get("costo_estimado_usd"),
                "phoenix_costo_usd": ph["costo_usd"],
                "api_tokens": f"{r.get('tokens_entrada')}/{r.get('tokens_salida')}",
                "phoenix_tokens": f"{ph['tokens_entrada']:.0f}/{ph['tokens_salida']:.0f}",
                "api_llamadas": r.get("llamadas_llm"),
                "phoenix_llamadas": int(ph["llamadas_llm"]),
            }
        )
    return filas


def a_markdown(rep: dict[str, Any], cruce: list[dict[str, Any]]) -> str:
    lat = rep["latencia_ms"]
    lineas = [
        f"# Costo, tokens y latencia según Phoenix (proyecto `{rep['proyecto']}`)",
        "",
        f"- Leído de la API GraphQL de Phoenix el {rep['fecha']}. Trazas: {rep['trazas_en_proyecto']}.",
        f"- Costo total: US$ {rep['costo_total_usd']:.6f} · tokens: {rep['tokens_totales']:.0f}",
        f"- Latencia de las trazas: p50 {(lat['p50'] or 0) / 1000:.1f} s · p95 {(lat['p95'] or 0) / 1000:.1f} s",
        f"- Capa RAG: {rep['rag']['spans_retriever']} spans RETRIEVER, latencia media {rep['rag']['latencia_media_ms']} ms "
        f"(máx. {rep['rag']['latencia_max_ms']} ms).",
        "",
        "## Por traza",
        "",
        "| inicio (UTC) | traza | job | conversación | fase | estado | latencia (s) | LLM | tokens in/out | costo (US$) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in rep["trazas"]:
        lineas.append(
            f"| {t['inicio'][11:19]} | {t['nombre']} | `{(t['job_id'] or '-')[:8]}` | `{t['thread_id'] or '-'}` | {t['fase'] or '-'} "
            f"| {t['estado']} | {t['latencia_s']} | {t['llamadas_llm']} | {t['tokens_entrada']}/{t['tokens_salida']} | {t['costo_usd']:.6f} |"
        )
    lineas += [
        "",
        "## Por nodo del grafo",
        "",
        "| nodo | llamadas LLM | tokens in/out | costo (US$) |",
        "|---|---|---|---|",
    ]
    for nodo, d in sorted(rep["por_nodo"].items(), key=lambda kv: -kv[1].get("costo_usd", 0)):
        lineas.append(
            f"| {nodo} | {d.get('llamadas_llm', 0):.0f} | {d.get('tokens_entrada', 0):.0f}/{d.get('tokens_salida', 0):.0f} | {d.get('costo_usd', 0):.6f} |"
        )
    if cruce:
        lineas += [
            "",
            "## Control cruzado: métricas de la API vs. Phoenix (mismo job)",
            "",
            "| escenario | job | costo API (US$) | costo Phoenix (US$) | tokens API | tokens Phoenix | LLM API | LLM Phoenix |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in cruce:
            lineas.append(
                f"| {f['escenario']} | `{f['job_id'][:8]}` | {f['api_costo_usd']} | {f['phoenix_costo_usd']:.6f} | {f['api_tokens']} "
                f"| {f['phoenix_tokens']} | {f['api_llamadas']} | {f['phoenix_llamadas']} |"
            )
    return "\n".join(lineas) + "\n"


def a_html(rep: dict[str, Any], cruce: list[dict[str, Any]]) -> str:
    lat = rep["latencia_ms"]
    filas = "".join(
        f"<tr><td>{t['nombre']}</td><td><code>{t['thread_id'] or '-'}</code></td><td>{t['fase'] or '-'}</td><td>{t['estado']}</td>"
        f"<td>{t['latencia_s']}</td><td>{t['llamadas_llm']}</td><td>{t['tokens_entrada']:,} / {t['tokens_salida']:,}</td>"
        f"<td><b>US$ {t['costo_usd']:.6f}</b></td></tr>"
        for t in rep["trazas"]
    )
    cruces = "".join(
        f"<tr><td>{f['escenario']}</td><td>US$ {f['api_costo_usd']}</td><td>US$ {f['phoenix_costo_usd']:.6f}</td>"
        f"<td>{f['api_tokens']}</td><td>{f['phoenix_tokens']}</td></tr>"
        for f in cruce
    )
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Costo y latencia</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#1b1b1f;background:#fff}}
table{{border-collapse:collapse;margin:8px 0 20px;font-size:14px}}td,th{{border:1px solid #ccd;padding:5px 10px;text-align:left}}
th{{background:#eef}}.kpi{{display:inline-block;margin:0 18px 12px 0;padding:10px 16px;border:1px solid #ccd;border-radius:8px}}
.kpi b{{display:block;font-size:22px}}small{{color:#555}}</style></head><body>
<h2>Costo, tokens y latencia: proyecto <code>{rep["proyecto"]}</code> en Phoenix</h2>
<small>Leído de la API GraphQL de Phoenix (costSummary, latencyMsQuantile) el {rep["fecha"]}.</small>
<p><span class="kpi">Trazas<b>{rep["trazas_en_proyecto"]}</b></span>
<span class="kpi">Costo total<b>US$ {rep["costo_total_usd"]:.6f}</b></span>
<span class="kpi">Tokens<b>{rep["tokens_totales"]:,.0f}</b></span>
<span class="kpi">Latencia p50 / p95<b>{(lat["p50"] or 0) / 1000:.1f} s / {(lat["p95"] or 0) / 1000:.1f} s</b></span>
<span class="kpi">Spans RETRIEVER (RAG)<b>{rep["rag"]["spans_retriever"]} · {rep["rag"]["latencia_media_ms"]} ms</b></span></p>
<h3>Por traza</h3>
<table><tr><th>traza</th><th>conversación</th><th>fase</th><th>estado</th><th>latencia (s)</th><th>LLM</th><th>tokens in / out</th><th>costo</th></tr>{filas}</table>
<h3>Métricas de la API vs. Phoenix (mismo job)</h3>
<table><tr><th>escenario</th><th>costo API</th><th>costo Phoenix</th><th>tokens API</th><th>tokens Phoenix</th></tr>{cruces}</table>
</body></html>"""


async def main() -> None:
    from app.core.config import get_settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:6006")
    parser.add_argument("--proyecto", default=None, help="por defecto, PHOENIX_PROJECT_NAME")
    args = parser.parse_args()
    rep = await analizar(args.url, args.proyecto or get_settings().phoenix_project_name)
    cruce = control_cruzado(rep)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "phoenix_report.json").write_text(
        json.dumps({**rep, "control_cruzado": cruce}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS / "phoenix_report.md").write_text(a_markdown(rep, cruce), encoding="utf-8")
    (DOCS / "phoenix_report.html").write_text(a_html(rep, cruce), encoding="utf-8")
    print(a_markdown(rep, cruce))


if __name__ == "__main__":
    asyncio.run(main())
