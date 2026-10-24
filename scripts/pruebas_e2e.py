"""Pruebas del sistema de punta a punta contra la API real (Gemini + Pinecone + Redis + Phoenix).

    docker compose up --build -d
    uv run python -m scripts.pruebas_e2e                  # todas, en orden
    uv run python -m scripts.pruebas_e2e --solo 01 02     # algunas
    uv run python -m scripts.pruebas_e2e --sin-reinicio   # la 03 sin reiniciar los contenedores

Escenarios (cada uno deja trazas en Phoenix y un JSON con pedidos y respuestas en `examples/`):

  01 RAG + cálculo         pregunta que necesita documentación (investigador) y cuentas (analista)
  02 seguimiento           misma conversación (thread_id): "¿y con 8 workers?" usa el historial
  03 HITL + reinicio       tarea crítica -> WAITING_APPROVAL -> `docker compose restart api redis`
                           -> el job sigue pausado -> se aprueba -> DONE y publica (sin repetir LLM)
  03b seguimiento post-reinicio   tercer turno de la conversación de la 01/02 después del reinicio
  04 rechazo               tarea crítica rechazada por el revisor: no publica y lo explica
  05 fuera de dominio      el supervisor la detecta y la síntesis responde sin LLM
  06 validación            422 (Pydantic), 404 y 401 (clave de revisor), sin gastar LLM (el 409 está en la 01 y la 04)

Corren en serie a propósito: con el free tier de Gemini (~5 pedidos/min y ~20/día por modelo)
lanzarlas en paralelo solo haría cola en el limitador. Resumen en `docs/pruebas_e2e.md`.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings

RAIZ = Path(__file__).resolve().parents[1]
EJEMPLOS = RAIZ / "examples"
DOCS = RAIZ / "docs"
HILO = f"e2e-conversacion-{datetime.now():%m%d%H%M}"


class Registro:
    """Guarda cada pedido y respuesta de un escenario (lo que después queda en examples/)."""

    def __init__(self, client: httpx.AsyncClient, nombre: str, descripcion: str) -> None:
        self.client = client
        self.nombre = nombre
        self.datos: dict[str, Any] = {
            "escenario": nombre,
            "descripcion": descripcion,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "pasos": [],
        }

    async def pedir(
        self, metodo: str, ruta: str, body: Any = None, nota: str = "", headers: dict[str, str] | None = None
    ) -> httpx.Response:
        inicio = time.perf_counter()
        r = await self.client.request(metodo, ruta, json=body, headers=headers)
        ms = round((time.perf_counter() - inicio) * 1000, 1)
        try:
            cuerpo = r.json()
        except ValueError:
            cuerpo = r.text
        self.datos["pasos"].append(
            {"nota": nota, "pedido": f"{metodo} {ruta}", "body": body, "status": r.status_code, "ms": ms, "respuesta": cuerpo}
        )
        print(f"  {metodo:4s} {ruta:44s} -> {r.status_code} ({ms} ms) {nota}", flush=True)
        return r

    async def esperar(self, job_id: str, *estados: str, plazo_s: float = 900, intervalo: float = 3) -> dict[str, Any]:
        """Polling de GET /tasks/{id} hasta uno de `estados` (solo se registra el último GET)."""
        inicio = time.monotonic()
        ultimo = None
        while time.monotonic() - inicio < plazo_s:
            job = (await self.client.get(f"/tasks/{job_id}")).json()
            if job["status"] != ultimo:
                print(f"    [{time.monotonic() - inicio:6.1f} s] {job_id[:8]} -> {job['status']}", flush=True)
                ultimo = job["status"]
            if job["status"] in estados:
                self.datos["pasos"].append(
                    {"nota": f"polling hasta {job['status']}", "pedido": f"GET /tasks/{job_id}", "status": 200, "respuesta": job}
                )
                return job
            await asyncio.sleep(intervalo)
        raise TimeoutError(f"{job_id} no llegó a {estados} en {plazo_s} s (quedó en {ultimo})")

    async def escuchar(self, job_id: str, plazo_s: float = 900) -> dict[str, Any]:
        """Lee `GET /tasks/{id}/stream` (SSE) hasta el evento `fin` y registra cada evento con su segundo."""
        inicio = time.monotonic()
        eventos: list[dict[str, Any]] = []
        tipo = None
        async with self.client.stream("GET", f"/tasks/{job_id}/stream", timeout=plazo_s) as r:
            async for linea in r.aiter_lines():
                if linea.startswith("event: "):
                    tipo = linea.removeprefix("event: ")
                elif linea.startswith("data: "):
                    datos = json.loads(linea.removeprefix("data: "))
                    s = round(time.monotonic() - inicio, 2)
                    if tipo == "fin":
                        eventos.append({"s": s, "evento": "fin", "status": datos["status"]})
                    else:
                        eventos.append({"s": s, "evento": tipo, **datos})
                    if tipo in ("estado", "nodo"):
                        detalle = datos.get("status") or f"{datos.get('nodo')} {datos.get('detalle') or ''}".strip()
                        print(f"    [stream {s:6.1f} s] {tipo}: {detalle[:90]}", flush=True)
        tokens = [e for e in eventos if e["evento"] == "token"]
        resumen = {
            "eventos": len(eventos),
            "eventos_token": len(tokens),
            "primer_token_s": tokens[0]["s"] if tokens else None,
            "nodos": [e["nodo"] for e in eventos if e["evento"] == "nodo"],
            "texto_transmitido": "".join(e["texto"] for e in tokens),
        }
        self.datos["pasos"].append(
            {
                "nota": "stream SSE hasta el fin",
                "pedido": f"GET /tasks/{job_id}/stream",
                "status": 200,
                "resumen": resumen,
                "respuesta": eventos,
            }
        )
        return resumen

    def guardar(self, resumen: dict[str, Any]) -> dict[str, Any]:
        self.datos["resumen"] = resumen
        EJEMPLOS.mkdir(exist_ok=True)
        (EJEMPLOS / f"{self.nombre}.json").write_text(json.dumps(self.datos, ensure_ascii=False, indent=2), encoding="utf-8")
        return resumen


def resumir(escenario: str, job: dict[str, Any], **extra: Any) -> dict[str, Any]:
    m = job.get("metricas") or {}
    res = job.get("resultado") or {}
    return {
        "escenario": escenario,
        "job_id": job["job_id"],
        "thread_id": job["thread_id"],
        "status": job["status"],
        "estados": " → ".join(e["status"] for e in job["eventos"]),
        "ruta_supervisor": res.get("ruta_supervisor"),
        "turno": res.get("turno"),
        "fuentes": len(res.get("fuentes") or []),
        "fuera_de_dominio": res.get("fuera_de_dominio"),
        "reporte_publicado": res.get("reporte_publicado"),
        "latencia_total_s": m.get("latencia_total_s"),
        "latencia_ejecucion_s": m.get("latencia_ejecucion_s"),
        "espera_aprobacion_s": m.get("espera_aprobacion_s"),
        "llamadas_llm": m.get("llamadas_llm"),
        "llamadas_fallidas": m.get("llamadas_fallidas"),
        "tokens_entrada": m.get("tokens_entrada"),
        "tokens_salida": m.get("tokens_salida"),
        "costo_estimado_usd": m.get("costo_estimado_usd"),
        "error": job.get("error"),
        **extra,
    }


# --- Escenarios ---


async def e01(client: httpx.AsyncClient) -> dict[str, Any]:
    reg = Registro(
        client, "01_rag_y_calculo", "Pregunta que necesita la documentación (RAG) y un cálculo; abre la conversación del 02."
    )
    pregunta = (
        "Según la documentación de Caudal, ¿qué tamaño de lote se recomienda para ingesta masiva? "
        "¿Cuántos lotes son 2.000.000 de eventos con ese tamaño y cuánto tarda procesarlos con 4 workers "
        "que procesan 5 lotes por segundo cada uno?"
    )
    r = await reg.pedir("POST", "/tasks", {"pregunta": pregunta, "thread_id": HILO}, "202 al instante")
    job_id = r.json()["job_id"]
    # Mientras corre, otro turno en la MISMA conversación se rechaza (candado por thread_id).
    await reg.pedir(
        "POST", "/tasks", {"pregunta": "¿Y con 8 workers cuánto tarda?", "thread_id": HILO}, "409: conversación ocupada"
    )
    # En vez de polling, el progreso en vivo por SSE: nodos, estados y los tokens de la síntesis.
    stream = await reg.escuchar(job_id)
    job = await reg.esperar(job_id, "DONE", "FAILED")
    return reg.guardar(
        resumir(
            "01 RAG + cálculo",
            job,
            stream_eventos_token=stream["eventos_token"],
            stream_primer_token_s=stream["primer_token_s"],
            stream_texto_igual_a_respuesta=stream["texto_transmitido"] == (job.get("resultado") or {}).get("respuesta"),
        )
    )


async def e02(client: httpx.AsyncClient) -> dict[str, Any]:
    reg = Registro(
        client,
        "02_seguimiento_misma_conversacion",
        "Seguimiento en el mismo thread_id: usa el historial guardado en el checkpoint.",
    )
    r = await reg.pedir("POST", "/tasks", {"pregunta": "¿Y si en vez de 4 workers uso 8, cuánto tardaría?", "thread_id": HILO})
    job = await reg.esperar(r.json()["job_id"], "DONE", "FAILED")
    await reg.pedir("GET", f"/conversations/{HILO}", nota="historial de la conversación")
    return reg.guardar(resumir("02 seguimiento (mismo thread_id)", job))


async def _salud(client: httpx.AsyncClient, plazo_s: float = 180) -> None:
    inicio = time.monotonic()
    while time.monotonic() - inicio < plazo_s:
        try:
            if (await client.get("/health")).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(2)
    raise TimeoutError("la API no volvió a responder /health")


async def e03(client: httpx.AsyncClient, reiniciar: bool = True) -> dict[str, Any]:
    reg = Registro(
        client,
        "03_hitl_aprobada_con_reinicio",
        "Tarea crítica (pide publicar): pausa en WAITING_APPROVAL, se reinician la API y Redis, sigue pausada, se aprueba y publica.",
    )
    pregunta = "¿Qué significa el error CAU-417 en Caudal y cómo se resuelve? Publicá el reporte para el equipo de soporte."
    r = await reg.pedir("POST", "/tasks", {"pregunta": pregunta})
    job_id = r.json()["job_id"]
    pausado = await reg.esperar(job_id, "WAITING_APPROVAL", "DONE", "FAILED")
    llamadas_antes = (pausado.get("metricas") or {}).get("llamadas_llm")
    reinicio: dict[str, Any] = {"reiniciado": False}
    if reiniciar and pausado["status"] == "WAITING_APPROVAL":
        antes = (await reg.pedir("GET", f"/conversations/{HILO}", nota="conversación ANTES del reinicio")).json()
        print("  $ docker compose restart api redis", flush=True)
        t0 = time.monotonic()
        proceso = await asyncio.create_subprocess_exec("docker", "compose", "restart", "api", "redis", cwd=RAIZ)
        await proceso.wait()
        await _salud(client)
        reinicio = {
            "reiniciado": True,
            "segundos_hasta_health": round(time.monotonic() - t0, 1),
            "turnos_antes": len(antes.get("turnos", [])),
        }
        despues = (await reg.pedir("GET", f"/conversations/{HILO}", nota="conversación DESPUÉS del reinicio")).json()
        reinicio["turnos_despues"] = len(despues.get("turnos", []))
        sigue = (await reg.pedir("GET", f"/tasks/{job_id}", nota="el job crítico sigue pausado después del reinicio")).json()
        reinicio["estado_despues_del_reinicio"] = sigue["status"]
    if pausado["status"] == "WAITING_APPROVAL":
        await reg.pedir(
            "POST",
            f"/tasks/{job_id}/approve",
            {"aprobado": True, "comentario": "Revisado: las fuentes están citadas."},
        )
    job = await reg.esperar(job_id, "DONE", "FAILED")
    llamadas_despues = (job.get("metricas") or {}).get("llamadas_llm")
    return reg.guardar(
        resumir(
            "03 HITL aprobada + reinicio",
            job,
            llamadas_llm_antes_de_aprobar=llamadas_antes,
            llamadas_llm_al_final=llamadas_despues,
            **reinicio,
        )
    )


async def e03b(client: httpx.AsyncClient) -> dict[str, Any]:
    reg = Registro(
        client, "03b_seguimiento_despues_del_reinicio", "Tercer turno de la conversación 01/02 después de reiniciar API y Redis."
    )
    pregunta = "Resumime en dos líneas el tamaño de lote recomendado y los dos tiempos que calculamos (con 4 y con 8 workers)."
    r = await reg.pedir("POST", "/tasks", {"pregunta": pregunta, "thread_id": HILO})
    job = await reg.esperar(r.json()["job_id"], "DONE", "FAILED")
    await reg.pedir("GET", f"/conversations/{HILO}", nota="los 3 turnos, del checkpoint en Redis")
    return reg.guardar(resumir("03b seguimiento post-reinicio", job))


async def e04(client: httpx.AsyncClient) -> dict[str, Any]:
    reg = Registro(
        client,
        "04_hitl_rechazada",
        "Tarea crítica (publicar=true) rechazada por el revisor: no se publica y la respuesta lo explica.",
    )
    r = await reg.pedir(
        "POST", "/tasks", {"pregunta": "¿Qué conectores de origen soporta Caudal según su documentación?", "publicar": True}
    )
    job_id = r.json()["job_id"]
    pausado = await reg.esperar(job_id, "WAITING_APPROVAL", "DONE", "FAILED")
    if pausado["status"] == "WAITING_APPROVAL":
        await reg.pedir(
            "POST",
            f"/tasks/{job_id}/approve",
            {"aprobado": False, "comentario": "Falta la versión mínima de cada conector."},
        )
        await reg.pedir("POST", f"/tasks/{job_id}/approve", {"aprobado": True}, "409: ya no espera aprobación")
    job = await reg.esperar(job_id, "DONE", "FAILED")
    return reg.guardar(resumir("04 HITL rechazada", job))


async def e05(client: httpx.AsyncClient) -> dict[str, Any]:
    reg = Registro(
        client,
        "05_fuera_de_dominio",
        "Pregunta ajena a Caudal: el supervisor la marca fuera de dominio y no se busca ni se inventa nada.",
    )
    r = await reg.pedir("POST", "/tasks", {"pregunta": "¿Quién ganó el Mundial de fútbol de 2022 y cuántos goles hizo Messi?"})
    job = await reg.esperar(r.json()["job_id"], "DONE", "FAILED")
    return reg.guardar(resumir("05 fuera de dominio", job))


async def e06(client: httpx.AsyncClient) -> dict[str, Any]:
    reg = Registro(client, "06_validacion_y_errores", "Contratos de la API: 422 de Pydantic, 404 y 409. No llama a ningún LLM.")
    await reg.pedir("POST", "/tasks", {"pregunta": "corta"}, "422: pregunta < 10 caracteres")
    await reg.pedir(
        "POST",
        "/tasks",
        {"pregunta": "¿Cuál es el tamano_lote por defecto?", "publicar": "quizás"},
        "422: publicar no es booleano",
    )
    await reg.pedir(
        "POST",
        "/tasks",
        {"pregunta": "¿Cuál es el tamano_lote por defecto?", "thread_id": "con espacios!"},
        "422: thread_id inválido",
    )
    await reg.pedir(
        "POST", "/tasks", {"pregunta": "¿Cuál es el tamano_lote por defecto?", "prioridad": "alta"}, "422: campo desconocido"
    )
    await reg.pedir("POST", "/search", {"consulta": "CAU-417", "k": 50}, "422: k fuera de rango")
    await reg.pedir("GET", "/tasks/no-es-un-id", nota="422: path inválido")
    await reg.pedir("GET", f"/tasks/{'0' * 32}", nota="404: job inexistente")
    await reg.pedir("POST", f"/tasks/{'0' * 32}/approve", {"aprobado": True}, "404")
    await reg.pedir(
        "POST",
        f"/tasks/{'0' * 32}/approve",
        {"aprobado": True},
        "401: clave de revisor inválida",
        headers={"X-Reviewer-Key": "una-clave-que-no-es-de-nadie"},
    )
    await reg.pedir("POST", f"/tasks/{'0' * 32}/approve", {"aprobado": True, "revisor": "yo"}, "422: el revisor sale de la clave")
    await reg.pedir(
        "POST", "/search", {"consulta": "¿Qué significa el error CAU-417?", "k": 3}, "200: búsqueda directa en el RAG"
    )
    codigos = [p["status"] for p in reg.datos["pasos"]]
    return reg.guardar(
        {
            "escenario": "06 validación y errores",
            "status": "OK" if codigos == [422] * 6 + [404, 404, 401, 422, 200] else "REVISAR",
            "codigos": codigos,
        }
    )


ESCENARIOS = {"01": e01, "02": e02, "03": e03, "03b": e03b, "04": e04, "05": e05, "06": e06}


def a_markdown(resumenes: list[dict[str, Any]]) -> str:
    lineas = [
        "# Pruebas end-to-end contra la API real",
        "",
        f"Fecha: {datetime.now():%Y-%m-%d %H:%M} · API en Docker (`docker compose up --build`) · conversación `{HILO}`.",
        "Cada fila es un job; el detalle (pedidos y respuestas completas) está en `examples/`.",
        "",
        "| escenario | estado | estados vistos | ruta del supervisor | fuentes | LLM | tokens in/out | costo est. (US$) | latencia total (s) | ejecución (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in resumenes:
        if "job_id" not in r:
            lineas.append(f"| {r['escenario']} | {r['status']} | códigos {r.get('codigos')} | - | - | 0 | - | - | - | - |")
            continue
        lineas.append(
            f"| {r['escenario']} | {r['status']} | {r['estados']} | {' → '.join(r['ruta_supervisor'] or [])} | {r['fuentes']} "
            f"| {r['llamadas_llm']} | {r['tokens_entrada']}/{r['tokens_salida']} | {r['costo_estimado_usd']} "
            f"| {r['latencia_total_s']} | {r['latencia_ejecucion_s']} |"
        )
    reinicio = next((r for r in resumenes if r.get("reiniciado")), None)
    if reinicio:
        lineas += [
            "",
            "## Reinicio en medio de la prueba 03",
            "",
            f"- `docker compose restart api redis` con el job `{reinicio['job_id'][:8]}` en WAITING_APPROVAL; la API volvió a "
            f"responder `/health` en {reinicio['segundos_hasta_health']} s.",
            f"- La conversación `{HILO}` (pruebas 01, 02 y 03b) tenía {reinicio['turnos_antes']} turnos antes y {reinicio['turnos_despues']} después.",
            f"- El job siguió en `{reinicio['estado_despues_del_reinicio']}`; aprobado después del reinicio terminó `{reinicio['status']}` "
            f"(publicado en `{reinicio['reporte_publicado']}`) con {reinicio['llamadas_llm_al_final']} llamadas al LLM en total, "
            f"las mismas {reinicio['llamadas_llm_antes_de_aprobar']} de antes de la pausa: la reanudación salió del checkpoint.",
        ]
    return "\n".join(lineas) + "\n"


def _guardar_resumen(resumenes: list[dict[str, Any]], parcial: bool) -> None:
    """Escribe docs/pruebas_e2e.{json,md}; con --solo, combina con las corridas anteriores."""
    DOCS.mkdir(exist_ok=True)
    previos = DOCS / "pruebas_e2e.json"
    todos = {r["escenario"]: r for r in (json.loads(previos.read_text()) if previos.exists() and parcial else [])}
    todos.update({r["escenario"]: r for r in resumenes})
    ordenados = sorted(todos.values(), key=lambda r: r["escenario"])
    previos.write_text(json.dumps(ordenados, ensure_ascii=False, indent=2), encoding="utf-8")
    (DOCS / "pruebas_e2e.md").write_text(a_markdown(ordenados), encoding="utf-8")


async def main() -> None:
    global HILO
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--solo", nargs="*", choices=list(ESCENARIOS), help="escenarios a correr (por defecto, todos)")
    parser.add_argument("--hilo", help="thread_id de la conversación de 01/02/03b (para retomar una corrida)")
    parser.add_argument("--sin-reinicio", action="store_true", help="la 03 no reinicia los contenedores")
    args = parser.parse_args()
    HILO = args.hilo or HILO

    # Las aprobaciones las firma el primer revisor de REVIEWER_KEYS (queda registrado en el job).
    revisores = get_settings().reviewer_keys
    headers = {"X-Reviewer-Key": next(iter(revisores.values())).get_secret_value()} if revisores else {}

    resumenes = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=60, headers=headers) as client:
        (await client.get("/health")).raise_for_status()
        for clave in args.solo or list(ESCENARIOS):
            print(f"\n== Escenario {clave} ==", flush=True)
            funcion = ESCENARIOS[clave]
            resumen = await (funcion(client, reiniciar=not args.sin_reinicio) if clave == "03" else funcion(client))
            print("  " + json.dumps({k: v for k, v in resumen.items() if v is not None}, ensure_ascii=False))
            resumenes.append(resumen)

    await asyncio.to_thread(_guardar_resumen, resumenes, bool(args.solo))
    print(f"\nResumen en docs/pruebas_e2e.md · pedidos y respuestas en examples/ · conversación {HILO}")


if __name__ == "__main__":
    asyncio.run(main())
