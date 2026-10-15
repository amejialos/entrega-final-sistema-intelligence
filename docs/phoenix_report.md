# Costo, tokens y latencia según Phoenix (proyecto `entrega-final`)

- Leído de la API GraphQL de Phoenix el 2026-10-15T08:26:18. Trazas: 12.
- Costo total: US$ 0.019914 · tokens: 45936
- Latencia de las trazas: p50 22.4 s · p95 87.9 s
- Capa RAG: 10 spans RETRIEVER, latencia media 1858.0 ms (máx. 2380.3 ms).

## Por traza

| inicio (UTC) | traza | job | conversación | fase | estado | latencia (s) | LLM | tokens in/out | costo (US$) |
|---|---|---|---|---|---|---|---|---|---|
| 11:07:19 | recuperacion_hibrida | `-` | `-` | - | OK | 2.33 | 0 | 0/0 | 0.000000 |
| 11:11:09 | recuperacion_hibrida | `-` | `-` | - | OK | 1.41 | 0 | 0/0 | 0.000000 |
| 11:11:16 | orquestador job=1c1ba872 (ejecutar) | `1c1ba872` | `e2e-conversacion-final` | ejecutar | OK | 112.66 | 9 | 10691/1029 | 0.005213 |
| 11:13:26 | orquestador job=a953a40c (ejecutar) | `a953a40c` | `e2e-conversacion-final` | ejecutar | OK | 34.7 | 5 | 5012/566 | 0.002526 |
| 11:14:02 | orquestador job=48790389 (ejecutar) | `48790389` | `2608a82de5bc42c2b41f3c7881f1aa72` | ejecutar | OK | 51.71 | 7 | 12015/569 | 0.004836 |
| 11:15:03 | orquestador job=48790389 (reanudar) | `48790389` | `2608a82de5bc42c2b41f3c7881f1aa72` | reanudar | OK | 0.01 | 0 | 0/0 | 0.000000 |
| 11:15:06 | orquestador job=3049339b (ejecutar) | `3049339b` | `e2e-conversacion-final` | ejecutar | OK | 34.09 | 2 | 1840/213 | 0.000970 |
| 11:15:42 | orquestador job=2798d055 (ejecutar) | `2798d055` | `eda658cffb134e7285b760bff6d1c337` | ejecutar | OK | 50.56 | 6 | 7163/472 | 0.003214 |
| 11:16:34 | orquestador job=2798d055 (reanudar) | `2798d055` | `eda658cffb134e7285b760bff6d1c337` | reanudar | OK | 0.02 | 0 | 0/0 | 0.000000 |
| 11:16:37 | orquestador job=ff691bff (ejecutar) | `ff691bff` | `b39a68157adb43e0bfa8225660d991ef` | ejecutar | OK | 10.74 | 1 | 703/102 | 0.000466 |
| 11:24:24 | orquestador job=ae73baec (ejecutar) | `ae73baec` | `a8939856ede14dfabc8c2b31bda40c6b` | ejecutar | OK | 67.67 | 5 | 5018/543 | 0.002690 |
| 11:25:42 | orquestador job=ae73baec (reanudar) | `ae73baec` | `a8939856ede14dfabc8c2b31bda40c6b` | reanudar | OK | 0.05 | 0 | 0/0 | 0.000000 |

## Por nodo del grafo

| nodo | llamadas LLM | tokens in/out | costo (US$) |
|---|---|---|---|
| researcher | 12 | 23079/538 | 0.008269 |
| supervisor | 13 | 11835/1779 | 0.007998 |
| synthesizer | 6 | 3503/842 | 0.002139 |
| analyst | 4 | 4025/335 | 0.001509 |

## Control cruzado: métricas de la API vs. Phoenix (mismo job)

| escenario | job | costo API (US$) | costo Phoenix (US$) | tokens API | tokens Phoenix | LLM API | LLM Phoenix |
|---|---|---|---|---|---|---|---|
| 01 RAG + cálculo | `1c1ba872` | 0.00521255 | 0.005213 | 10691/1029 | 10691/1029 | 9 | 9 |
| 02 seguimiento (mismo thread_id) | `a953a40c` | 0.00252565 | 0.002526 | 5012/566 | 5012/566 | 5 | 5 |
| 03 HITL aprobada + reinicio | `ae73baec` | 0.0026897 | 0.002690 | 5018/543 | 5018/543 | 5 | 5 |
| 03b seguimiento post-reinicio | `3049339b` | 0.00097025 | 0.000970 | 1840/213 | 1840/213 | 2 | 2 |
| 04 HITL rechazada | `2798d055` | 0.00321425 | 0.003214 | 7163/472 | 7163/472 | 6 | 6 |
| 05 fuera de dominio | `ff691bff` | 0.0004659 | 0.000466 | 703/102 | 703/102 | 1 | 1 |
