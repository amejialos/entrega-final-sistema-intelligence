"""La API: 202 + job_id sin esperar al grafo, polling PENDING -> RUNNING -> DONE, FAILED ante una
excepción, validación Pydantic (422) de bodies y paths, concurrencia y robustez de la cola."""

import asyncio
import json
import time

import pytest

from app.worker.models import Estado
from app.worker.worker import Worker
from tests.conftest import PREGUNTA, grafo_simple, levantar_api


async def test_post_devuelve_202_y_job_id_sin_esperar_al_grafo():
    compuerta = asyncio.Event()  # el grafo no termina hasta que la abramos
    async with levantar_api(grafo_simple(compuerta)) as api:
        inicio = time.monotonic()
        r = await api.client.post("/tasks", json={"pregunta": PREGUNTA})
        demora = time.monotonic() - inicio

        assert r.status_code == 202
        cuerpo = r.json()
        assert len(cuerpo["job_id"]) == 32 and cuerpo["status"] == "PENDING"
        assert len(cuerpo["thread_id"]) == 32  # sin thread_id se abre una conversación nueva
        assert cuerpo["links"] == {
            "estado": f"/tasks/{cuerpo['job_id']}",
            "conversacion": f"/conversations/{cuerpo['thread_id']}",
            "aprobar": None,
        }
        assert r.headers["location"] == f"/tasks/{cuerpo['job_id']}"
        assert demora < 0.5  # respondió con el grafo todavía corriendo

        job = await api.esperar(cuerpo["job_id"], "RUNNING")
        assert job["started_at"] and job["resultado"] is None

        compuerta.set()
        job = await api.esperar(cuerpo["job_id"], "DONE")
        assert job["resultado"]["respuesta"] == f"respuesta a: {PREGUNTA}"
        assert [e["status"] for e in job["eventos"]] == ["PENDING", "RUNNING", "DONE"]
        assert job["finished_at"] >= job["started_at"] >= job["created_at"]


async def test_excepcion_en_el_worker_deja_el_job_failed_con_el_error():
    async with levantar_api(grafo_simple(falla=ConnectionError("Pinecone no responde"))) as api:
        job_id = await api.crear()
        job = await api.esperar(job_id, "FAILED", "DONE")

        assert job["status"] == "FAILED"
        assert job["error"] == "ConnectionError: Pinecone no responde"
        assert job["finished_at"] and job["resultado"] is None
        assert job["metricas"]["latencia_total_s"] >= 0  # también hay métricas de un job fallido
        assert [e["status"] for e in job["eventos"]] == ["PENDING", "RUNNING", "FAILED"]
        # El worker sigue vivo y la conversación quedó libre para otro turno.
        assert api.worker._tareas and not any(t.done() for t in api.worker._tareas)
        assert await api.store.job_activo(job["thread_id"]) is None


async def test_cinco_jobs_corren_en_paralelo_sin_bloquear_la_api():
    compuerta = asyncio.Event()
    async with levantar_api(grafo_simple(compuerta), concurrencia=5) as api:
        ids = await asyncio.gather(*(api.crear(f"Pregunta número {i} sobre Caudal") for i in range(5)))
        # Los 5 llegan a RUNNING a la vez (ninguno espera a que termine otro)...
        await asyncio.gather(*(api.esperar(i, "RUNNING") for i in ids))
        # ...y mientras tanto la API sigue respondiendo.
        assert (await api.client.get("/health")).status_code == 200
        compuerta.set()
        jobs = await asyncio.gather(*(api.esperar(i, "DONE") for i in ids))
        assert {j["resultado"]["respuesta"] for j in jobs} == {f"respuesta a: Pregunta número {i} sobre Caudal" for i in range(5)}
        listado = (await api.client.get("/tasks", params={"limite": 10})).json()
        assert {j["job_id"] for j in listado} == set(ids)


@pytest.mark.parametrize(
    "body",
    [
        {},  # falta la pregunta
        {"pregunta": "corta"},  # menos de 10 caracteres
        {"pregunta": " " * 20},  # solo espacios
        {"pregunta": "x" * 2001},  # demasiado larga
        {"pregunta": PREGUNTA, "publicar": "quizás"},  # no es booleano
        {"pregunta": PREGUNTA, "publicar": 1},  # StrictBool: 1 no es true
        {"pregunta": 123456789012},  # no es texto
        {"pregunta": PREGUNTA, "thread_id": "con espacios!"},  # thread_id fuera del patrón
        {"pregunta": PREGUNTA, "thread_id": "abc"},  # thread_id demasiado corto
        {"pregunta": PREGUNTA, "prioridad": "alta"},  # campo desconocido (extra="forbid")
    ],
)
async def test_body_invalido_responde_422_y_no_crea_jobs(body):
    async with levantar_api(grafo_simple()) as api:
        r = await api.client.post("/tasks", json=body)
        assert r.status_code == 422
        assert r.json()["detail"][0]["loc"][0] == "body"
        assert await api.store.listar() == []


async def test_paths_invalidos_responden_422_y_jobs_inexistentes_404():
    async with levantar_api(grafo_simple()) as api:
        assert (await api.client.get("/tasks/no-es-un-id")).status_code == 422
        assert (await api.client.get(f"/tasks/{'0' * 32}")).status_code == 404
        r = await api.client.post(f"/tasks/{'0' * 32}/approve", json={"aprobado": True})
        assert r.status_code == 404
        assert (await api.client.get("/conversations/no valido")).status_code == 422
        assert (await api.client.get("/conversations/no-existe-todavia")).status_code == 404


async def test_el_estado_del_job_vive_en_redis():
    compuerta = asyncio.Event()
    async with levantar_api(grafo_simple(compuerta)) as api:
        job_id = await api.crear()
        await api.esperar(job_id, "RUNNING")
        crudo = await api.redis.hgetall(f"final:job:{job_id}")
        assert crudo["status"] == Estado.RUNNING and crudo["pregunta"].startswith("¿Cuál")
        compuerta.set()
        await api.esperar(job_id, "DONE")
        assert json.loads(await api.redis.hget(f"final:job:{job_id}", "resultado"))["task_completed"] is True


async def test_recuperar_reencola_mensajes_de_un_worker_muerto():
    """Un proceso murió con un mensaje tomado (su lista `procesando` quedó sin señal de vida)."""
    async with levantar_api(grafo_simple()) as api:
        await api.worker.detener()
        job = await api.store.crear(PREGUNTA)
        await api.redis.lpush("final:procesando:muerto", json.dumps({"tipo": "ejecutar", "job_id": job.job_id}))
        # formato viejo (una sola lista compartida): también se recupera
        viejo = await api.store.crear("otra pregunta")
        await api.redis.lpush("final:procesando", json.dumps({"tipo": "ejecutar", "job_id": viejo.job_id}))

        await api.worker.iniciar()  # al arrancar, recuperar() los devuelve a la cola

        assert (await api.esperar(job.job_id, "DONE"))["resultado"]["task_completed"] is True
        assert (await api.esperar(viejo.job_id, "DONE"))["status"] == "DONE"
        assert await api.redis.exists("final:procesando:muerto", "final:procesando") == 0


async def test_un_worker_que_arranca_no_le_roba_mensajes_a_otro_que_esta_vivo():
    """Con varios procesos (WORKER_EMBEBIDO=0), recuperar() solo toca las listas de workers sin latido."""
    compuerta = asyncio.Event()
    async with levantar_api(grafo_simple(compuerta)) as api:
        job_id = await api.crear()
        await api.esperar(job_id, "RUNNING")  # el worker A lo tiene tomado y está vivo
        otro = Worker(api.redis, api.store, grafo_simple(), espera_cola_s=0.05)
        assert await otro.recuperar() == 0  # B arranca: no toca la lista de A
        assert await api.redis.llen(api.worker.procesando) == 1

        await api.redis.delete(api.worker.clave_latido)  # A "muere": su latido vence
        api.worker._activo = False
        for tarea in api.worker._tareas:  # sin dejar que A siga (ni renueve el latido)
            tarea.cancel()
        assert await otro.recuperar() == 1  # ahora sí: el mensaje vuelve a la cola
        assert await api.redis.llen("final:cola") == 1
        compuerta.set()


async def test_un_mensaje_invalido_en_la_cola_se_descarta_sin_matar_al_worker():
    async with levantar_api(grafo_simple()) as api:
        await api.redis.lpush("final:cola", json.dumps({"tipo": "borrar_todo", "job_id": "x"}))
        await api.redis.lpush("final:cola", "esto no es json")
        job_id = await api.crear()  # el siguiente job se procesa normalmente
        assert (await api.esperar(job_id, "DONE"))["status"] == "DONE"
        assert await api.redis.llen(api.worker.procesando) == 0


async def test_health_y_openapi_documentan_los_contratos():
    async with levantar_api(grafo_simple()) as api:
        salud = (await api.client.get("/health")).json()
        assert salud["status"] == "ok" and salud["redis"] == "ok"
        assert salud["rag"] == {"indice": "indice-test", "namespace": "ns-test", "chunks_bm25": 21}
        esquema = (await api.client.get("/openapi.json")).json()
        post = esquema["paths"]["/tasks"]["post"]
        assert {"202", "409", "422"} <= set(post["responses"])
        assert post["responses"]["202"]["content"]["application/json"]["schema"]["$ref"].endswith("TareaAceptada")
        assert esquema["paths"]["/tasks/{job_id}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ].endswith("JobRecord")
        for ruta in ("/tasks/{job_id}/approve", "/conversations/{thread_id}", "/search", "/health"):
            assert ruta in esquema["paths"]
        assert "Metricas" in esquema["components"]["schemas"] and "Fuente" in esquema["components"]["schemas"]


async def test_health_responde_503_con_contrato_si_redis_no_responde():
    from redis.exceptions import ConnectionError as RedisConnectionError

    async with levantar_api(grafo_simple()) as api:

        async def caido():
            raise RedisConnectionError("Redis caído")

        api.app.state.recursos.redis.ping = caido
        r = await api.client.get("/health")
        assert r.status_code == 503 and r.json() == {"detail": "Redis no responde."}
