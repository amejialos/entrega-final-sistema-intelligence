"""Prueba mínima de cada modelo configurado (un pedido por modelo) antes de gastar cuota en serio.

    uv run python -m scripts.probar_modelos                       # los modelos del .env (uno por rol)
    uv run python -m scripts.probar_modelos --extra gemini-flash-latest gemini-3-flash-preview

Con el free tier de Gemini (~20 pedidos por día POR MODELO) conviene saber qué modelos responden
hoy antes de correr las pruebas end-to-end. No imprime keys: solo modelo, estado, latencia y tokens.
"""

import argparse
import asyncio
import time

from langchain_core.messages import HumanMessage

from app.core.config import get_settings
from app.core.providers import ChatOpenAICompatible


async def probar(modelo: str) -> str:
    settings = get_settings()
    llm = ChatOpenAICompatible(
        model=modelo, api_key=settings.openai_api_key, base_url=settings.openai_base_url, temperature=0, max_retries=0, timeout=60
    )
    inicio = time.perf_counter()
    try:
        respuesta = await llm.ainvoke([HumanMessage("Respondé solamente con la palabra: ok")])
    except Exception as exc:
        codigo = getattr(exc, "status_code", "?")
        return f"{modelo:32s} ERROR {codigo}  {str(exc)[:140]}"
    uso = respuesta.usage_metadata or {}
    return (
        f"{modelo:32s} OK   {time.perf_counter() - inicio:5.1f} s  "
        f"tokens {uso.get('input_tokens', '?')}/{uso.get('output_tokens', '?')}  -> {respuesta.text.strip()[:30]!r}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extra", nargs="*", default=[], help="otros modelos para probar")
    args = parser.parse_args()
    modelos = list(dict.fromkeys([*get_settings().modelos().values(), *args.extra]))
    for linea in await asyncio.gather(*(probar(m) for m in modelos)):
        print(linea)


if __name__ == "__main__":
    asyncio.run(main())
