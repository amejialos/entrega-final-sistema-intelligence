# Evidencia en Redis (después de las pruebas end-to-end)

Generado con `scripts/evidencia_redis.sh` contra el contenedor `redis` del compose el 2026-10-15 08:26, después del reinicio de la prueba 03.

```text
$ redis-cli INFO persistence | grep aof_enabled
aof_enabled:1
$ redis-cli MODULE LIST   # JSON + Search, que necesita AsyncRedisSaver
search
ReJSON
$ redis-cli FT._LIST   # índices de RediSearch creados por asetup()
checkpoint_write
checkpoint
$ redis-cli --scan --pattern 'checkpoint:e2e-conversacion-final:*' | wc -l   # checkpoints de la conversación
41
$ redis-cli --scan --pattern 'checkpoint_write:e2e-conversacion-final:*' | wc -l
151
$ redis-cli LRANGE final:thread:e2e-conversacion-final:jobs 0 -1   # los turnos (jobs) de la conversación
1c1ba8726ecc4d70923b07cdd9255cbb
a953a40cd4324adeaaa65e13c5ed98ea
3049339b876241e284b0ad8f25bc3be1
$ redis-cli EXISTS final:thread:e2e-conversacion-final:activo   # 0 = el candado se liberó al terminar
0
$ redis-cli ZCARD final:jobs
7
$ redis-cli LRANGE final:reportes_publicados 0 -1   # efectos secundarios aprobados
ae73baec47c24a749e4c413af9cb8a37
487903891cda493e9e9201b2906edb29
$ redis-cli HGET final:job:ae73baec47c24a749e4c413af9cb8a37 status
DONE
$ redis-cli LRANGE final:job:ae73baec47c24a749e4c413af9cb8a37:eventos 0 -1   # historial del job crítico
{"status": "PENDING", "t": "2026-10-15T11:24:24.113+00:00"}
{"status": "RUNNING", "t": "2026-10-15T11:24:24.132+00:00"}
{"status": "WAITING_APPROVAL", "t": "2026-10-15T11:25:31.838+00:00"}
{"status": "PENDING", "t": "2026-10-15T11:25:42.241+00:00", "por": "approve"}
{"status": "RUNNING", "t": "2026-10-15T11:25:42.250+00:00"}
{"status": "DONE", "t": "2026-10-15T11:25:42.322+00:00"}
```
