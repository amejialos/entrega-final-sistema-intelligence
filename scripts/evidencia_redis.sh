#!/usr/bin/env bash
# Evidencia de persistencia en Redis: AOF, módulos, checkpoints de una conversación, jobs y candados.
#   ./scripts/evidencia_redis.sh <thread_id> <job_id_critico>  > docs/redis_evidencia.md
set -euo pipefail
cd "$(dirname "$0")/.."
HILO="${1:?thread_id de la conversación}"
JOB="${2:?job_id de la tarea crítica}"
R() { docker compose exec -T redis redis-cli "$@" | tr -d '\r'; }
echo "# Evidencia en Redis (después de las pruebas end-to-end)"
echo
echo "Generado con \`scripts/evidencia_redis.sh\` contra el contenedor \`redis\` del compose el $(date '+%Y-%m-%d %H:%M'), después del reinicio de la prueba 03."
echo
echo '```text'
echo '$ redis-cli INFO persistence | grep aof_enabled'; R INFO persistence | grep -E "^aof_enabled"
echo '$ redis-cli MODULE LIST   # JSON + Search, que necesita AsyncRedisSaver'; R MODULE LIST | grep -E "^(search|ReJSON)$"
echo "\$ redis-cli FT._LIST   # índices de RediSearch creados por asetup()"; R FT._LIST
echo "\$ redis-cli --scan --pattern 'checkpoint:$HILO:*' | wc -l   # checkpoints de la conversación"; R --scan --pattern "checkpoint:$HILO:*" | wc -l | tr -d ' '
echo "\$ redis-cli --scan --pattern 'checkpoint_write:$HILO:*' | wc -l"; R --scan --pattern "checkpoint_write:$HILO:*" | wc -l | tr -d ' '
echo "\$ redis-cli LRANGE final:thread:$HILO:jobs 0 -1   # los turnos (jobs) de la conversación"; R LRANGE "final:thread:$HILO:jobs" 0 -1
echo "\$ redis-cli EXISTS final:thread:$HILO:activo   # 0 = el candado se liberó al terminar"; R EXISTS "final:thread:$HILO:activo"
echo '$ redis-cli ZCARD final:jobs'; R ZCARD final:jobs
echo '$ redis-cli LRANGE final:reportes_publicados 0 -1   # efectos secundarios aprobados'; R LRANGE final:reportes_publicados 0 -1
echo "\$ redis-cli HGET final:job:$JOB status"; R HGET "final:job:$JOB" status
echo "\$ redis-cli LRANGE final:job:$JOB:eventos 0 -1   # historial del job crítico"; R LRANGE "final:job:$JOB:eventos" 0 -1
echo '```'
