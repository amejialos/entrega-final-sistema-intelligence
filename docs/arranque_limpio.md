# Arranque en una máquina limpia (con una cuenta de Pinecone sin el índice)

Prueba del 2026-10-24, tal como la haría quien corrige: clonar desde GitHub en una carpeta nueva, copiar el
`.env.example` y completar **solo** las dos keys y la clave del revisor. `INDEX_NAME=sistema-intelligence` (el
valor del `.env.example`) **no existía** en la cuenta de Pinecone. Volúmenes de Docker vacíos (`-p limpio`).

```bash
git clone -b feature/sistema-intelligence https://github.com/amejialos/entrega-final-sistema-intelligence.git limpio
cd limpio && cp .env.example .env      # + OPENAI_API_KEY, PINECONE_API_KEY y REVIEWER_KEYS
docker compose -p limpio up --build -d --wait
#  Container limpio-redis-1   Healthy
#  Container limpio-phoenix-1 Healthy
#  Container limpio-api-1     Healthy          (53,7 s en total, build incluido)
```

**La API creó el índice y subió el corpus sola** (`docker compose logs api`):

```text
INFO app.rag.ingesta: El índice 'sistema-intelligence' no tiene el corpus: se ingesta ahora (una sola vez).
INFO app.rag.ingesta: Creando el índice serverless 'sistema-intelligence'...
INFO app.rag.ingesta: Ingesta automática: 21 chunks en 10.1 s.
INFO app.main: Sistema listo: modelos {...}, índice sistema-intelligence/caudal-docs
```

**Verificación:**

```bash
curl -s localhost:8000/health
# {"status":"ok","redis":"ok","version":"1.0.0","modelos":{...},"rag":{"indice":"sistema-intelligence",
#  "namespace":"caudal-docs","chunks_bm25":21},"trazas":"entrega-final","worker_concurrencia":5,
#  "aprobacion":"protegida","revisores":1}

curl -s -X POST localhost:8000/search -d '{"consulta":"¿Qué significa el error CAU-417?","k":3}' -H 'content-type: application/json'
# híbrido, 1451 ms: caudal-errores-p1-c0, caudal-errores-p2-c0, caudal-seguridad-p1-c0 (los tres por vector y BM25:
# la vía vectorial ya usa el índice recién creado)

# Una tarea completa, seguida por SSE:
#   eventos recibidos: 3 × estado, 4 × nodo, 1 × fin
#   GET /tasks/{id} -> DONE · pasos: supervisor, synthesizer, evaluar_criticidad, cerrar_turno · 1 llamada al LLM

curl -s -X POST localhost:8000/tasks/{id}/approve -H 'X-Reviewer-Key: x' -d '{"aprobado":true}' ...
# 401
```

Después: `docker compose -p limpio down -v` y se borró el `.env` copiado.
