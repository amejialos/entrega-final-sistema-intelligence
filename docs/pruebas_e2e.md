# Pruebas end-to-end contra la API real

Fecha: 2026-10-24 11:46 · API en Docker (`docker compose up --build`) · conversación `e2e-conversacion-final-10241141`.
Cada fila es un job; el detalle (pedidos y respuestas completas) está en `examples/`.

| escenario | estado | estados vistos | ruta del supervisor | fuentes | LLM | tokens in/out | costo est. (US$) | latencia total (s) | ejecución (s) |
|---|---|---|---|---|---|---|---|---|---|
| 01 RAG + cálculo | DONE | PENDING → RUNNING → DONE | researcher → analyst → FINISH | 5 | 10 | 15282/1879 | 0.00754165 | 206.075 | 206.067 |
| 02 seguimiento (mismo thread_id) | DONE | PENDING → RUNNING → DONE | analyst → FINISH | 0 | 5 | 7299/648 | 0.0031928 | 45.302 | 45.285 |
| 03 HITL aprobada + reinicio | DONE | PENDING → RUNNING → WAITING_APPROVAL → PENDING → RUNNING → DONE | researcher → FINISH | 5 | 5 | 8838/1264 | 0.0047451 | 44.638 | 38.287 |
| 03b seguimiento post-reinicio | DONE | PENDING → RUNNING → DONE | FINISH | 0 | 2 | 3944/321 | 0.00164695 | 37.567 | 37.56 |
| 04 HITL rechazada | DONE | PENDING → RUNNING → WAITING_APPROVAL → PENDING → RUNNING → DONE | researcher → FINISH | 7 | 6 | 13806/1325 | 0.0063605 | 63.33 | 60.961 |
| 05 fuera de dominio | DONE | PENDING → RUNNING → DONE | FINISH | 0 | 1 | 703/115 | 0.0004984 | 6.454 | 6.439 |
| 06 validación y errores | OK | códigos [422, 422, 422, 422, 422, 422, 404, 404, 401, 422, 200] | - | - | 0 | - | - | - | - |

## Reinicio en medio de la prueba 03

- `docker compose restart api redis` con el job `4b6e1c6c` en WAITING_APPROVAL; la API volvió a responder `/health` en 5.2 s.
- La conversación `e2e-conversacion-final-10241141` (pruebas 01, 02 y 03b) tenía 1 turnos antes y 1 después.
- El job siguió en `WAITING_APPROVAL`; aprobado después del reinicio terminó `DONE` (publicado en `final:reporte:4b6e1c6c30c7450fa436100d1938614f`) con 5 llamadas al LLM en total, las mismas 5 de antes de la pausa: la reanudación salió del checkpoint.
