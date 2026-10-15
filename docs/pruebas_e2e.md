# Pruebas end-to-end contra la API real

Fecha: 2026-10-15 08:16 · API en Docker (`docker compose up --build`) · conversación `e2e-conversacion-final`.
Cada fila es un job; el detalle (pedidos y respuestas completas) está en `examples/`.

| escenario | estado | estados vistos | ruta del supervisor | fuentes | LLM | tokens in/out | costo est. (US$) | latencia total (s) | ejecución (s) |
|---|---|---|---|---|---|---|---|---|---|
| 01 RAG + cálculo | DONE | PENDING → RUNNING → DONE | researcher → analyst → FINISH | 6 | 9 | 10691/1029 | 0.00521255 | 112.678 | 112.667 |
| 02 seguimiento (mismo thread_id) | DONE | PENDING → RUNNING → DONE | analyst → FINISH | 0 | 5 | 5012/566 | 0.00252565 | 34.727 | 34.705 |
| 03 HITL aprobada + reinicio | DONE | PENDING → RUNNING → WAITING_APPROVAL → PENDING → RUNNING → DONE | researcher → FINISH | 7 | 7 | 12015/569 | 0.0048359 | 60.635 | 51.727 |
| 03b seguimiento post-reinicio | DONE | PENDING → RUNNING → DONE | FINISH | 0 | 2 | 1840/213 | 0.00097025 | 34.115 | 34.094 |
| 04 HITL rechazada | DONE | PENDING → RUNNING → WAITING_APPROVAL → PENDING → RUNNING → DONE | researcher → FINISH | 5 | 6 | 7163/472 | 0.00321425 | 51.37 | 50.587 |
| 05 fuera de dominio | DONE | PENDING → RUNNING → DONE | FINISH | 0 | 1 | 703/102 | 0.0004659 | 10.761 | 10.748 |
| 06 validación y errores | OK | códigos [422, 422, 422, 422, 422, 422, 404, 404, 200] | - | - | 0 | - | - | - | - |

## Reinicio en medio de la prueba 03

- `docker compose restart api redis` con el job `48790389` en WAITING_APPROVAL; la API volvió a responder `/health` en 6.0 s.
- La conversación de las pruebas 01/02 tenía 2 turnos antes y 2 después.
- El job siguió en `WAITING_APPROVAL`; aprobado después del reinicio terminó `DONE` (publicado en `final:reporte:487903891cda493e9e9201b2906edb29`) con 7 llamadas al LLM en total, las mismas 7 de antes de la pausa: la reanudación salió del checkpoint.
