#!/bin/bash
# Exact-engine worker pool. NW workers claim CH items at a time from a queue file QFILE (an integer: the next item index)
# under flock on QFILE.lock.
# Single machine (default): the queue is OUT/queue.next on this machine; no ssh.
# Several machines (optional): run pool.sh on each with the same ITEMS and QHOST=user@host (ssh target holding QFILE, a Linux
# host: macOS has no flock). Admission per claim: MemAvailable >= MEMMIN MB and load1 < LOADMAX; 10 failed chunks -> STOP.
# Runner: RUNNER=ens (default; warm ensemble runner) or emurun (one process per chunk); both give identical results.
# usage: ITEMS=f OUT=dir [QFILE=/abs/queue.next] [QHOST=user@host] [RUNNER=ens|emurun] [CPUS=2-11] [CH=3] pool.sh NW
#        touches OUT/DONE_<hostname>; stop early with `touch OUT/STOP`.
source "$(dirname "$0")/env.sh" || exit 1
NW=${1:?NW}; CH=${CH:-3}; RUNNER=${RUNNER:-ens}; MEMMIN=${MEMMIN:-4000}; LOADMAX=${LOADMAX:-12}
: "${ITEMS:?ITEMS}" "${OUT:?OUT}"; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); QFILE=${QFILE:-$OUT/queue.next}
ITEMS=$(cd "$(dirname "$ITEMS")" && pwd)/$(basename "$ITEMS"); N=$(grep -c . "$ITEMS"); mkdir -p "$OUT/logs"
Q() { if [ -n "$QHOST" ]; then ssh -o BatchMode=yes -o ConnectTimeout=20 "$QHOST" "$1"; else bash -c "$1"; fi; }
[ -n "$QHOST" ] || [ -f "$QFILE" ] || echo 0 > "$QFILE"
claim() { while true; do
    r=$(Q "flock $QFILE.lock bash -c 'n=\$(cat $QFILE); [ \$n -ge $N ] && exit 3; echo \$((n+$CH)) > $QFILE; echo \$n'"); rc=$?
    [ $rc -eq 0 ] && { echo $r; return 0; }; [ $rc -eq 3 ] && return 3; sleep 60; done; }
room() { if [ "$(uname)" = Darwin ]; then
    vm_stat | awk -v M=$MEMMIN '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{exit !((f+i+s)*ps/1048576 >= M)}' && sysctl -n vm.loadavg | awk -v L=$LOADMAX '{exit !($2 < L)}'
  else m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); l=$(cut -d' ' -f1 /proc/loadavg); awk -v m=$m -v l=$l -v M=$MEMMIN -v L=$LOADMAX 'BEGIN{exit !(m>=M && l<L)}'; fi; }
pin() { if [ -n "$CPUS" ] && command -v taskset >/dev/null; then taskset -c $CPUS "$@"; else "$@"; fi; }
worker() { h=$1; sleep $((h*20))
  while true; do
    [ -f "$OUT/STOP" ] && break
    room || { sleep 60; continue; }
    s=$(claim) || break; e=$((s+CH-1)); [ $e -ge $N ] && e=$((N-1)); idx=$(seq $s $e | paste -sd, -)
    [[ $idx =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "BADIDX $idx" >> "$OUT/logs/err.log"; touch "$OUT/STOP"; break; }
    echo "$idx $(date +%T)" >> "$OUT/logs/claims_$h.txt"
    if [ $RUNNER = ens ]; then
      pin nice -n 15 $PY -m incognita_emu.runner.ens --out "$OUT" --grid $G --items "$ITEMS" --idx $s:$((e+1)) >> "$OUT/logs/w$h.log" 2>&1
    else
      pin nice -n 15 $PY -m incognita_emu.runner.emurun --out "$OUT" --grid $G --items "$ITEMS" --idx $idx >> "$OUT/logs/w$h.log" 2>&1
    fi || { echo "rc=$? $idx" >> "$OUT/logs/err.log"; [ $(grep -c . "$OUT/logs/err.log") -ge 10 ] && touch "$OUT/STOP"; }
  done; }
for ((h=0; h<NW; h++)); do worker $h & done; wait
touch "$OUT/DONE_$(hostname -s)"
