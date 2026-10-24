# Costo, tokens y latencia según Phoenix (proyecto `entrega-final`)

- Leído de la API GraphQL de Phoenix el 2026-10-24T11:47:55. Trazas: 12.
- Costo total: US$ 0.034344 · tokens: 80203
- Latencia de las trazas: p50 41.8 s · p95 181.1 s
- Capa RAG: 13 spans RETRIEVER, latencia media 2022.7 ms (máx. 2397.7 ms).

## Por traza

| inicio (UTC) | traza | job | conversación | fase | estado | latencia (s) | LLM | tokens in/out | costo (US$) |
|---|---|---|---|---|---|---|---|---|---|
| 14:31:45 | orquestador job=09fbd43c (ejecutar) | `09fbd43c` | `e2e-conversacion-10241131` | ejecutar | ERROR | 130.2 | 6 | 1569/193 | 0.000953 |
| 14:33:55 | orquestador job=3f242877 (ejecutar) | `3f242877` | `e2e-conversacion-10241131` | ejecutar | OK | 98.83 | 7 | 10243/627 | 0.004223 |
| 14:35:35 | orquestador job=4b6e1c6c (ejecutar) | `4b6e1c6c` | `2fe39ffe4bea4502b2f6a29f5c2b07c8` | ejecutar | OK | 38.27 | 5 | 8838/1264 | 0.004745 |
| 14:36:19 | orquestador job=4b6e1c6c (reanudar) | `4b6e1c6c` | `2fe39ffe4bea4502b2f6a29f5c2b07c8` | reanudar | OK | 0.02 | 0 | 0/0 | 0.000000 |
| 14:36:22 | orquestador job=33de2bfb (ejecutar) | `33de2bfb` | `e2e-conversacion-10241131` | ejecutar | OK | 160.75 | 10 | 11159/988 | 0.005183 |
| 14:39:05 | orquestador job=9ad45dca (ejecutar) | `9ad45dca` | `73c720accc00417b90763064c498b7fc` | ejecutar | OK | 60.93 | 6 | 13806/1325 | 0.006360 |
| 14:40:09 | orquestador job=9ad45dca (reanudar) | `9ad45dca` | `73c720accc00417b90763064c498b7fc` | reanudar | OK | 0.03 | 0 | 0/0 | 0.000000 |
| 14:40:12 | orquestador job=8ebaf5e6 (ejecutar) | `8ebaf5e6` | `4c905971bde34414b805800244ac34e8` | ejecutar | OK | 6.44 | 1 | 703/115 | 0.000498 |
| 14:40:21 | recuperacion_hibrida | `-` | `-` | - | OK | 2.15 | 0 | 0/0 | 0.000000 |
| 14:41:13 | orquestador job=0e35dc6f (ejecutar) | `0e35dc6f` | `e2e-conversacion-final-10241141` | ejecutar | OK | 206.06 | 12 | 15282/1879 | 0.007542 |
| 14:44:39 | orquestador job=d1437cce (ejecutar) | `d1437cce` | `e2e-conversacion-final-10241141` | ejecutar | OK | 45.28 | 5 | 7299/648 | 0.003193 |
| 14:45:52 | orquestador job=36d8def1 (ejecutar) | `36d8def1` | `e2e-conversacion-final-10241141` | ejecutar | OK | 37.56 | 2 | 3944/321 | 0.001647 |

## Por nodo del grafo

| nodo | llamadas LLM | tokens in/out | costo (US$) |
|---|---|---|---|
| synthesizer | 7 | 28866/3765 | 0.012864 |
| supervisor | 19 | 15254/2400 | 0.010576 |
| researcher | 23 | 24120/725 | 0.009048 |
| analyst | 5 | 4603/470 | 0.001856 |

## Control cruzado: métricas de la API vs. Phoenix (mismo job)

Las llamadas que fallaron (p. ej. un 503 de alta demanda que la `RetryPolicy` reintentó) quedan en
Phoenix como spans LLM en ERROR con 0 tokens: no cambian el costo ni los tokens. La API las cuenta
en `metricas.llamadas_fallidas`.

| escenario | job | costo API (US$) | costo Phoenix (US$) | tokens API | tokens Phoenix | LLM API (ok) | LLM Phoenix (ok + en ERROR) |
|---|---|---|---|---|---|---|---|
| 01 RAG + cálculo | `0e35dc6f` | 0.00754165 | 0.007542 | 15282/1879 | 15282/1879 | 10 | 10 + 2 |
| 02 seguimiento (mismo thread_id) | `d1437cce` | 0.0031928 | 0.003193 | 7299/648 | 7299/648 | 5 | 5 + 0 |
| 03 HITL aprobada + reinicio | `4b6e1c6c` | 0.0047451 | 0.004745 | 8838/1264 | 8838/1264 | 5 | 5 + 0 |
| 03b seguimiento post-reinicio | `36d8def1` | 0.00164695 | 0.001647 | 3944/321 | 3944/321 | 2 | 2 + 0 |
| 04 HITL rechazada | `9ad45dca` | 0.0063605 | 0.006360 | 13806/1325 | 13806/1325 | 6 | 6 + 0 |
| 05 fuera de dominio | `8ebaf5e6` | 0.0004984 | 0.000498 | 703/115 | 703/115 | 1 | 1 + 0 |
