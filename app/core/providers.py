"""Modelos de chat (uno por rol) y embeddings, armados a partir de `Settings`.

Se usa `ChatOpenAI` contra cualquier endpoint compatible con la API de OpenAI: con una key de
OpenAI anda tal cual, y con `OPENAI_BASE_URL` apuntando al endpoint compatible de Gemini se
prueba gratis. Las llamadas son asíncronas (`ainvoke` usa el cliente `AsyncOpenAI` de httpx).

Cada rol puede usar un modelo distinto (`SUPERVISOR_MODEL`, `RESEARCH_MODEL`, `ANALYST_MODEL`,
`SYNTH_MODEL`): sirve para repartir la cuota del free tier de Gemini, que se cuenta por modelo.

Detalle de Gemini (viene de la Pre-entrega 5): los modelos con razonamiento devuelven en cada
tool call un campo extra, `extra_content.google.thought_signature`, y exigen recibirlo de vuelta
en el turno siguiente (si falta: 400 "Function call is missing a thought_signature").
`ChatOpenAI` descarta ese campo; `ChatOpenAICompatible` lo guarda en el AIMessage y lo reinyecta
al armar el pedido. Con OpenAI el campo no existe y la subclase no cambia nada.
"""

from typing import Any

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from app.core.config import Rol, Settings

# Clave de additional_kwargs donde se guarda {tool_call_id: extra_content}.
EXTRA_CONTENT_KEY = "tool_call_extra_content"


class ChatOpenAICompatible(ChatOpenAI):
    """ChatOpenAI que preserva el `extra_content` de las tool calls (thought signatures de Gemini)."""

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for choice, generation in zip(data.get("choices") or [], result.generations, strict=False):
            extras = {
                tc["id"]: tc["extra_content"]
                for tc in (choice.get("message") or {}).get("tool_calls") or []
                if tc.get("extra_content") and tc.get("id")
            }
            if extras and isinstance(generation.message, AIMessage):
                generation.message.additional_kwargs[EXTRA_CONTENT_KEY] = extras
        return result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        # En chat/completions hay un dict por mensaje, en el mismo orden.
        for message, message_dict in zip(messages, payload.get("messages") or [], strict=False):
            extras = message.additional_kwargs.get(EXTRA_CONTENT_KEY) if isinstance(message, AIMessage) else None
            for tool_call in message_dict.get("tool_calls") or [] if extras else []:
                if tool_call.get("id") in extras:
                    tool_call["extra_content"] = extras[tool_call["id"]]
        return payload


# Un limitador POR MODELO: la cuota por minuto del free tier se cuenta por modelo, y con varios
# jobs concurrentes los roles que comparten modelo tienen que compartir el límite.
_LIMITADORES: dict[str, InMemoryRateLimiter] = {}


def _rate_limiter(modelo: str, rpm: float | None) -> InMemoryRateLimiter | None:
    if not rpm:
        return None
    rps = rpm / 60
    limitador = _LIMITADORES.get(modelo)
    if limitador is None or limitador.requests_per_second != rps:
        limitador = InMemoryRateLimiter(requests_per_second=rps, check_every_n_seconds=0.5, max_bucket_size=1)
        _LIMITADORES[modelo] = limitador
    return limitador


def build_model(settings: Settings, rol: Rol) -> ChatOpenAICompatible:
    """Modelo del rol con la key, el endpoint y el límite de pedidos por minuto de `settings`."""
    modelo = settings.modelo_para(rol)
    return ChatOpenAICompatible(
        model=modelo,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,  # orquestación y cálculo: lo más determinista posible
        # Sin reintentos del SDK: los hace el nodo del grafo (RetryPolicy) con esperas largas.
        # Los reintentos rápidos del SDK contra un 429 solo gastan más cuota.
        max_retries=0,
        timeout=settings.llm_timeout_s,
        rate_limiter=_rate_limiter(modelo, settings.llm_rpm),
        disable_streaming=True,  # el rescate de extra_content está en la ruta sin streaming
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """`OpenAIEmbeddings` contra OpenAI o contra el endpoint compatible de Gemini (async con `aembed_*`)."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimension,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        # Con un endpoint que no es OpenAI no se tokeniza con tiktoken antes de mandar: el
        # endpoint de Gemini espera texto, no listas de ids de tokens.
        check_embedding_ctx_length=settings.openai_base_url is None,
        chunk_size=100,  # el endpoint de Gemini acepta como mucho 100 textos por pedido
        max_retries=2,
    )
