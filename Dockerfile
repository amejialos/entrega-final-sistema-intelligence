# Imagen de la API (incluye el worker embebido). La levanta `docker compose up --build`.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/srv/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/srv/.tiktoken

# Primero solo las dependencias (capa cacheada mientras no cambie uv.lock).
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev
# El tokenizador de tiktoken se descarga en el build: el chunking del arranque (índice BM25)
# no depende de tener internet hacia el blob de OpenAI.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY data ./data
# Los scripts operativos (ingesta, pruebas, reporte) también se pueden correr dentro del contenedor:
#   docker compose exec api python -m scripts.ingest --solo-indice
COPY scripts ./scripts

# Sin root dentro del contenedor.
RUN useradd --create-home --uid 1000 app && chown -R app /srv
USER app

EXPOSE 8000
# Un solo proceso: la API y el worker embebido (WORKER_CONCURRENCY jobs a la vez).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
