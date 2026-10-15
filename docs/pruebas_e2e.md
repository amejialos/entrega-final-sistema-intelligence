# Pruebas end-to-end contra la API real

Fecha de generación: 2026-10-15 08:27 · API en Docker (`docker compose up --build`) · conversación `e2e-conversacion-final`.
Cada fila es un job; el detalle (pedidos y respuestas completas) está en `examples/`.

| escenario | estado | estados vistos | ruta del supervisor | fuentes | LLM | tokens in/out | costo est. (US$) | latencia total (s) | ejecución (s) |
|---|---|---|---|---|---|---|---|---|---|
| 01 RAG + cálculo | DONE | PENDING → RUNNING → DONE | researcher → analyst → FINISH | 6 | 9 | 10691/1029 | 0.00521255 | 112.678 | 112.667 |
| 02 seguimiento (mismo thread_id) | DONE | PENDING → RUNNING → DONE | analyst → FINISH | 0 | 5 | 5012/566 | 0.00252565 | 34.727 | 34.705 |
| 03 HITL aprobada + reinicio | DONE | PENDING → RUNNING → WAITING_APPROVAL → PENDING → RUNNING → DONE | researcher → FINISH | 5 | 5 | 5018/543 | 0.0026897 | 78.209 | 67.746 |
| 03b seguimiento post-reinicio | DONE | PENDING → RUNNING → DONE | FINISH | 0 | 2 | 1840/213 | 0.00097025 | 34.115 | 34.094 |
| 04 HITL rechazada | DONE | PENDING → RUNNING → WAITING_APPROVAL → PENDING → RUNNING → DONE | researcher → FINISH | 5 | 6 | 7163/472 | 0.00321425 | 51.37 | 50.587 |
| 05 fuera de dominio | DONE | PENDING → RUNNING → DONE | FINISH | 0 | 1 | 703/102 | 0.0004659 | 10.761 | 10.748 |
| 06 validación y errores | OK | códigos [422, 422, 422, 422, 422, 422, 404, 404, 200] | - | - | 0 | - | - | - | - |

## Reinicio en medio de la prueba 03

- `docker compose restart api redis` con el job `ae73baec` en WAITING_APPROVAL; la API volvió a responder `/health` en 7.8 s.
- La conversación `e2e-conversacion-final` (pruebas 01, 02 y 03b) tenía 3 turnos antes y 3 después.
- El job siguió en `WAITING_APPROVAL`; aprobado después del reinicio terminó `DONE` (publicado en `final:reporte:ae73baec47c24a749e4c413af9cb8a37`) con 5 llamadas al LLM en total, las mismas 5 de antes de la pausa: la reanudación salió del checkpoint.

## Nota: la primera corrida de la 03 encontró un bug

La primera vez que corrió la prueba 03 (job `48790389`, su traza sigue en Phoenix), el investigador
respondió que el significado de CAU-417 "no figura en la documentación", aunque `/search` traía el
chunk correcto primero. Causa: la herramienta recortaba cada fragmento a 900 caracteres (herencia de
la PE7) y en ese chunk el dato empieza en el carácter 874. Se subió el tope a 2.000 (el chunk más
largo mide 1.962), se marca cualquier corte y se agregó un test de regresión
(`tests/test_tools.py::test_el_modelo_lee_los_chunks_completos_y_un_corte_queda_marcado`). La 03 se
volvió a correr: respuesta correcta y citada, con 5 llamadas al LLM en vez de 7 (ya no hizo falta
reformular la búsqueda).
