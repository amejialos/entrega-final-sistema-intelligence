"""Métricas por respuesta: latencia, llamadas al LLM, tokens y costo estimado de cada job.

Phoenix tiene estas métricas por traza, pero el cliente de la API no debería tener que abrir
Phoenix para saber cuánto tardó y cuánto costó SU respuesta. Por eso cada job devuelve un
bloque `metricas` (modelo Pydantic `Metricas`) en `GET /tasks/{id}`.

`MetricasLLM` es un callback de LangChain que se pasa en el `config` de cada corrida del grafo:
LangChain lo propaga a TODOS los modelos que se llaman adentro (supervisor, los agentes ReAct
de los especialistas, la síntesis), así que suma tokens sin tocar el código de los nodos.
Cada job tiene su propia instancia: con 5 jobs concurrentes las cuentas no se mezclan.

- Modelo: `ls_model_name` de la metadata que LangChain adjunta a cada llamada (el nombre
  configurado en el `.env`, el mismo que usa Phoenix para el precio).
- Tokens: `AIMessage.usage_metadata` (input/output), normalizado por LangChain para cualquier
  proveedor.
- Costo: tokens x precio por millón de `Settings.precios_llm`. Es una ESTIMACIÓN (con el free
  tier de Gemini el gasto real es 0).
"""

import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, Field

from app.core.config import PrecioModelo


class UsoModelo(BaseModel):
    llamadas: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    costo_estimado_usd: float = 0.0


class Metricas(BaseModel):
    """Lo que costó y tardó una respuesta. Se acumula entre la corrida inicial y la reanudación."""

    latencia_total_s: float | None = Field(
        default=None, description="De POST /tasks a DONE/FAILED: incluye la cola y la espera de aprobación."
    )
    latencia_ejecucion_s: float = Field(default=0.0, description="Tiempo corriendo el grafo (suma de corridas).")
    espera_aprobacion_s: float | None = Field(default=None, description="Tiempo en WAITING_APPROVAL.")
    corridas: int = Field(default=0, description="1 sin HITL; 2 con pausa y reanudación.")
    llamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    tokens_totales: int = 0
    costo_estimado_usd: float = 0.0
    por_modelo: dict[str, UsoModelo] = Field(default_factory=dict)
    modelos_sin_precio: list[str] = Field(default_factory=list)

    def sumar(self, otra: "Metricas") -> "Metricas":
        """Acumula otra corrida del mismo job (por ejemplo, la reanudación tras aprobar)."""
        por_modelo = {k: v.model_copy() for k, v in self.por_modelo.items()}
        for modelo, uso in otra.por_modelo.items():
            actual = por_modelo.setdefault(modelo, UsoModelo())
            actual.llamadas += uso.llamadas
            actual.tokens_entrada += uso.tokens_entrada
            actual.tokens_salida += uso.tokens_salida
            actual.costo_estimado_usd = round(actual.costo_estimado_usd + uso.costo_estimado_usd, 8)
        return Metricas(
            latencia_total_s=otra.latencia_total_s if otra.latencia_total_s is not None else self.latencia_total_s,
            latencia_ejecucion_s=round(self.latencia_ejecucion_s + otra.latencia_ejecucion_s, 3),
            espera_aprobacion_s=otra.espera_aprobacion_s if otra.espera_aprobacion_s is not None else self.espera_aprobacion_s,
            corridas=self.corridas + otra.corridas,
            llamadas_llm=self.llamadas_llm + otra.llamadas_llm,
            tokens_entrada=self.tokens_entrada + otra.tokens_entrada,
            tokens_salida=self.tokens_salida + otra.tokens_salida,
            tokens_totales=self.tokens_totales + otra.tokens_totales,
            costo_estimado_usd=round(self.costo_estimado_usd + otra.costo_estimado_usd, 8),
            por_modelo=por_modelo,
            modelos_sin_precio=sorted(set(self.modelos_sin_precio) | set(otra.modelos_sin_precio)),
        )


def normalizar_modelo(nombre: str) -> str:
    """`models/gemini-x` -> `gemini-x` (el endpoint de Gemini a veces antepone `models/`)."""
    return nombre.strip().lower().removeprefix("models/")


def costo_estimado(tokens_entrada: int, tokens_salida: int, precio: PrecioModelo) -> float:
    return (tokens_entrada * precio.entrada_usd_por_millon + tokens_salida * precio.salida_usd_por_millon) / 1_000_000


class MetricasLLM(BaseCallbackHandler):
    """Callback que cuenta llamadas y tokens por modelo durante UNA corrida del grafo."""

    # Corre en el event loop (no en un hilo aparte): solo suma enteros.
    run_inline = True

    def __init__(self, precios: dict[str, PrecioModelo]) -> None:
        super().__init__()
        self._precios = {normalizar_modelo(k): v for k, v in precios.items()}
        self._lock = threading.Lock()
        self._modelo_por_run: dict[UUID, str] = {}
        self.uso: dict[str, UsoModelo] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        nombre = (metadata or {}).get("ls_model_name")
        if nombre:
            self._modelo_por_run[run_id] = normalizar_modelo(str(nombre))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        generacion = response.generations[0][0] if response.generations and response.generations[0] else None
        mensaje = generacion.message if isinstance(generacion, ChatGeneration) else None
        usage = mensaje.usage_metadata if isinstance(mensaje, AIMessage) else None
        modelo = self._modelo_por_run.pop(run_id, None) or (
            normalizar_modelo(str(mensaje.response_metadata.get("model_name", "desconocido"))) if mensaje else "desconocido"
        )
        with self._lock:
            uso = self.uso.setdefault(modelo, UsoModelo())
            uso.llamadas += 1
            if usage:
                uso.tokens_entrada += int(usage.get("input_tokens", 0))
                uso.tokens_salida += int(usage.get("output_tokens", 0))

    def metricas(self, latencia_ejecucion_s: float) -> Metricas:
        """Resumen de la corrida, con el costo estimado por modelo."""
        por_modelo: dict[str, UsoModelo] = {}
        sin_precio = []
        for modelo, uso in self.uso.items():
            precio = self._precios.get(modelo)
            if precio is None:
                sin_precio.append(modelo)
            costo = costo_estimado(uso.tokens_entrada, uso.tokens_salida, precio) if precio else 0.0
            por_modelo[modelo] = uso.model_copy(update={"costo_estimado_usd": round(costo, 8)})
        entrada = sum(u.tokens_entrada for u in por_modelo.values())
        salida = sum(u.tokens_salida for u in por_modelo.values())
        return Metricas(
            latencia_ejecucion_s=round(latencia_ejecucion_s, 3),
            corridas=1,
            llamadas_llm=sum(u.llamadas for u in por_modelo.values()),
            tokens_entrada=entrada,
            tokens_salida=salida,
            tokens_totales=entrada + salida,
            costo_estimado_usd=round(sum(u.costo_estimado_usd for u in por_modelo.values()), 8),
            por_modelo=por_modelo,
            modelos_sin_precio=sorted(sin_precio),
        )
