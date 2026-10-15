# Evidencia en Redis (después de las pruebas end-to-end)

Comandos corridos contra el contenedor `redis` del compose el 2026-10-15 08:21, después del reinicio de la prueba 03.

```text
$ redis-cli INFO persistence | grep aof_enabled
aof_enabled:1
$ redis-cli MODULE LIST   # JSON + Search, que necesita AsyncRedisSaver
search
ReJSON
$ redis-cli --scan --pattern "checkpoint:e2e-conversacion-final:*" | wc -l   # checkpoints de la conversación de 3 turnos
41
$ redis-cli --scan --pattern "checkpoint_write:e2e-conversacion-final:*" | wc -l
151
$ redis-cli FT._LIST   # índices de RediSearch creados por asetup()
checkpoint_write
checkpoint
$ redis-cli LRANGE final:thread:e2e-conversacion-final:jobs 0 -1   # los 3 turnos (jobs) de la conversación
1c1ba8726ecc4d70923b07cdd9255cbb
a953a40cd4324adeaaa65e13c5ed98ea
3049339b876241e284b0ad8f25bc3be1
$ redis-cli EXISTS final:thread:e2e-conversacion-final:activo   # 0 = el candado se liberó al terminar
0
$ redis-cli ZCARD final:jobs
6
$ redis-cli LRANGE final:reportes_publicados 0 -1   # el efecto secundario aprobado (prueba 03); la 04 rechazada no publicó
487903891cda493e9e9201b2906edb29
$ redis-cli HGET final:job:487903891cda493e9e9201b2906edb29 status
DONE
$ redis-cli LRANGE final:job:487903891cda493e9e9201b2906edb29:eventos 0 -1   # historial del job crítico
{"status": "PENDING", "t": "2026-10-15T11:14:02.768+00:00"}
{"status": "RUNNING", "t": "2026-10-15T11:14:02.770+00:00"}
{"status": "WAITING_APPROVAL", "t": "2026-10-15T11:14:54.485+00:00"}
{"status": "PENDING", "t": "2026-10-15T11:15:03.387+00:00", "por": "approve"}
{"status": "RUNNING", "t": "2026-10-15T11:15:03.388+00:00"}
{"status": "DONE", "t": "2026-10-15T11:15:03.403+00:00"}
```
