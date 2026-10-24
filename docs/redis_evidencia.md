# Evidencia en Redis (después de las pruebas end-to-end)

Generado con `scripts/evidencia_redis.sh` contra el contenedor `redis` del compose el 2026-10-24 11:46, después del reinicio de la prueba 03.

```text
$ redis-cli INFO persistence | grep aof_enabled
aof_enabled:1
$ redis-cli MODULE LIST   # JSON + Search, que necesita AsyncRedisSaver
search
ReJSON
$ redis-cli FT._LIST   # índices de RediSearch creados por asetup()
checkpoint_write
checkpoint
$ redis-cli --scan --pattern 'checkpoint:e2e-conversacion-final-10241141:*' | wc -l   # checkpoints de la conversación
47
$ redis-cli --scan --pattern 'checkpoint_write:e2e-conversacion-final-10241141:*' | wc -l
158
$ redis-cli LRANGE final:thread:e2e-conversacion-final-10241141:jobs 0 -1   # los turnos (jobs) de la conversación
0e35dc6fb25b47c2ae39aad9639015d7
d1437cceb8874e10ad126dc79f8280d3
36d8def1425b4f77a84683f79c9e7500
$ redis-cli EXISTS final:thread:e2e-conversacion-final-10241141:activo   # 0 = el candado se liberó al terminar
0
$ redis-cli ZCARD final:jobs
9
$ redis-cli LRANGE final:reportes_publicados 0 -1   # efectos secundarios aprobados
4b6e1c6c30c7450fa436100d1938614f
$ redis-cli HGET final:job:4b6e1c6c30c7450fa436100d1938614f status
DONE
$ redis-cli LRANGE final:job:4b6e1c6c30c7450fa436100d1938614f:eventos 0 -1   # historial del job crítico
{"status": "PENDING", "t": "2026-10-24T14:35:35.175+00:00"}
{"status": "RUNNING", "t": "2026-10-24T14:35:35.177+00:00"}
{"status": "WAITING_APPROVAL", "t": "2026-10-24T14:36:13.450+00:00"}
{"status": "PENDING", "t": "2026-10-24T14:36:19.739+00:00", "por": "approve"}
{"status": "RUNNING", "t": "2026-10-24T14:36:19.790+00:00"}
{"status": "DONE", "t": "2026-10-24T14:36:19.813+00:00"}
```
