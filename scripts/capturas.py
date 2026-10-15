"""Capturas de Phoenix y de la API para el README (Playwright, Chromium headless).

    uv run --with playwright==1.56.0 python -m scripts.capturas
    CHROMIUM_PATH=/ruta/a/headless_shell uv run --with playwright==1.56.0 python -m scripts.capturas

Playwright no es dependencia del proyecto (solo se usa para documentar): `uv run --with` lo
instala en un entorno efímero. Si no se indica `CHROMIUM_PATH`, usa el Chromium propio de
Playwright (`uv run --with playwright==1.56.0 playwright install chromium`).

Toma las trazas de las pruebas end-to-end (`docs/pruebas_e2e.json`) y las busca en Phoenix por
la metadata `job_id`, así cada captura muestra la corrida que dice mostrar.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from playwright.async_api import Page, async_playwright

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "screenshots"
PHOENIX = os.environ.get("PHOENIX_URL", "http://localhost:6006")
API = os.environ.get("API_URL", "http://localhost:8000")


async def gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=PHOENIX, timeout=60) as client:
        r = await client.post("/graphql", json={"query": query, "variables": variables or {}})
        r.raise_for_status()
        return r.json()["data"]


async def datos_de_phoenix(proyecto: str) -> dict[str, Any]:
    proyectos = await gql("{ projects { edges { node { id name } } } }")
    pid = next(e["node"]["id"] for e in proyectos["projects"]["edges"] if e["node"]["name"] == proyecto)
    spans, cursor = [], None
    consulta = """query($id: ID!, $after: String) { node(id: $id) { ... on Project {
        spans(first: 500, after: $after) { pageInfo { hasNextPage endCursor }
          edges { node { id name spanKind parentId metadata context { traceId } } } } } } }"""
    while True:
        pagina = (await gql(consulta, {"id": pid, "after": cursor}))["node"]["spans"]
        spans += [e["node"] for e in pagina["edges"]]
        if not pagina["pageInfo"]["hasNextPage"]:
            break
        cursor = pagina["pageInfo"]["endCursor"]
    sesiones = await gql(
        "query($id: ID!) { node(id: $id) { ... on Project { sessions(first: 50) { edges { node { id sessionId } } } } } }",
        {"id": pid},
    )
    return {
        "pid": pid,
        "spans": spans,
        "sesiones": {e["node"]["sessionId"]: e["node"]["id"] for e in sesiones["node"]["sessions"]["edges"]},
    }


def traza_de(datos: dict[str, Any], job_id: str, fase: str = "ejecutar") -> str:
    for s in datos["spans"]:
        if s["parentId"] is None and s["metadata"]:
            meta = json.loads(s["metadata"]) if isinstance(s["metadata"], str) else s["metadata"]
            if meta.get("job_id") == job_id and meta.get("fase") == fase:
                return s["context"]["traceId"]
    raise LookupError(f"No encontré la traza del job {job_id} ({fase})")


def span_de(datos: dict[str, Any], trace_id: str, nombre: str, kind: str | None = None) -> str:
    for s in datos["spans"]:
        if s["context"]["traceId"] == trace_id and s["name"] == nombre and (kind is None or s["spanKind"].upper() == kind):
            return s["id"]
    raise LookupError(f"No encontré el span {nombre} en la traza {trace_id}")


async def capturar(page: Page, url: str, archivo: str, espera_ms: int = 3500, pestania: str | None = None) -> None:
    await page.goto(url)
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(espera_ms)
    if pestania:  # p. ej. la pestaña "Events" del span, donde queda el Interrupt
        await page.get_by_role("tab", name=pestania).first.click()
        await page.wait_for_timeout(1500)
    await page.screenshot(path=str(SALIDA / archivo))
    print(f"  {archivo}")


def pagina_de_job(job: dict[str, Any]) -> str:
    """HTML simple con la respuesta de GET /tasks/{id}: resultado, fuentes del RAG y métricas."""
    resultado = job["resultado"] or {}
    datos = {
        "job_id": job["job_id"],
        "thread_id": job["thread_id"],
        "status": job["status"],
        "resultado.respuesta": resultado.get("respuesta"),
        "resultado.fuentes (3 de " + str(len(resultado.get("fuentes") or [])) + ")": (resultado.get("fuentes") or [])[:3],
        "resultado.ruta_supervisor": resultado.get("ruta_supervisor"),
        "metricas": job["metricas"],
    }
    texto = json.dumps(datos, ensure_ascii=False, indent=2).replace("&", "&amp;").replace("<", "&lt;")
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>body{font-family:system-ui;margin:20px;background:#fff}"
        "pre{font-size:12.5px;background:#f6f8fa;padding:14px;border-radius:8px;white-space:pre-wrap}</style></head><body>"
        f"<h3>GET /tasks/{job['job_id']} (extracto de la respuesta real)</h3><pre>{texto}</pre></body></html>"
    )


async def main() -> None:
    from app.core.config import get_settings

    pruebas = {r["escenario"][:3].strip(): r for r in json.loads((RAIZ / "docs" / "pruebas_e2e.json").read_text())}
    datos = await datos_de_phoenix(get_settings().phoenix_project_name)
    pid = datos["pid"]
    rag = pruebas["01"]["job_id"]
    hitl = pruebas["03"]["job_id"]
    t_rag = traza_de(datos, rag)
    t_hitl = traza_de(datos, hitl)
    SALIDA.mkdir(exist_ok=True)

    async with async_playwright() as p:
        navegador = await p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = await navegador.new_page(viewport={"width": 1500, "height": 950})
        base = f"{PHOENIX}/projects/{pid}"
        print("Phoenix:")
        await capturar(page, f"{base}/traces", "01_trazas_lista.png")
        raiz = span_de(datos, t_rag, f"orquestador job={rag[:8]} (ejecutar)")
        await capturar(page, f"{base}/traces/{t_rag}?selectedSpanNodeId={raiz}", "02_traza_rag_y_calculo.png", 6000)
        await capturar(
            page,
            f"{base}/traces/{t_rag}?selectedSpanNodeId={span_de(datos, t_rag, 'recuperacion_hibrida', 'RETRIEVER')}",
            "03_span_rag_fuentes_scores.png",
        )
        await capturar(
            page,
            f"{base}/traces/{t_hitl}?selectedSpanNodeId={span_de(datos, t_hitl, 'aprobacion_humana')}",
            "04_hitl_pausa_interrupt.png",
            pestania="Events",
        )
        sesion = datos["sesiones"].get(pruebas["01"]["thread_id"])
        if sesion:
            await capturar(page, f"{base}/sessions/{sesion}", "05_sesion_conversacion_thread_id.png")
        llm = span_de(datos, t_rag, "ChatOpenAICompatible", "LLM")
        await capturar(page, f"{base}/traces/{t_rag}?selectedSpanNodeId={llm}", "06_span_llm_tokens_costo.png")
        await capturar(page, f"{base}/metrics", "07_metricas_costo_latencia.png", espera_ms=5000)
        print("Reportes y API:")
        await capturar(page, (RAIZ / "docs" / "phoenix_report.html").as_uri(), "08_costo_por_traza_y_control_cruzado.png", 1500)
        await capturar(page, f"{API}/docs", "09_api_swagger.png", 2500)
        async with httpx.AsyncClient(base_url=API) as client:
            job = (await client.get(f"/tasks/{rag}")).json()
        await page.set_content(pagina_de_job(job))
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(SALIDA / "10_respuesta_api_fuentes_y_metricas.png"), full_page=True)
        print("  10_respuesta_api_fuentes_y_metricas.png")
        await navegador.close()


if __name__ == "__main__":
    asyncio.run(main())
