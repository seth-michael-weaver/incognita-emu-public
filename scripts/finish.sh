#!/bin/bash
# Round finisher: waits for the queue to drain, STOPs every pool, waits for each pool's DONE_<host> file, re-runs locally any
# item that has no npz on any pool (SWEEPNW workers), extracts every pool's runs to $DATA/ext_<TAG>_<n>.npz and touches
# $DATA/<TAG>.DONE. Then fit/gate (README 'Training').
# Single machine (default): QHOST=local, QFILE=$OUT/queue.next, POOLS="local:$OUT".
# Several machines (optional): POOLS = space-separated "host:dir" (host "local" = this machine, else an ssh target),
# QHOST:QFILE = the shared queue; REMOTE_REPO = path of this repo on the remote hosts (default incognita-emu).
# usage: TAG=r1 ITEMS=items.txt OUT=runs/r1 DATA=data/ext finish.sh
source "$(dirname "$0")/env.sh" || exit 1
: "${TAG:?}" "${ITEMS:?}" "${DATA:?}"; QHOST=${QHOST:-local}
if [ -z "$POOLS" ]; then : "${OUT:?set OUT (single machine) or POOLS}"; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); POOLS="local:$OUT"; fi
QFILE=${QFILE:-${OUT:?set QFILE (multi-machine)}/queue.next}; mkdir -p $DATA; N=$(grep -c . $ITEMS); L=$DATA/finish_$TAG.log
R() { local h=$1; shift; if [ $h = local ]; then bash -c "$*"; else timeout 600 ssh -n -o BatchMode=yes $h "$*"; fi; }
log() { echo "$(date +%F' '%T) $*" >> $L; }
log start N=$N
while true; do q=$(R $QHOST cat $QFILE); [ -n "$q" ] && [ "$q" -ge $N ] && break; sleep 120; done
log drained; for p in $POOLS; do R ${p%%:*} touch ${p#*:}/STOP; done
for p in $POOLS; do until R ${p%%:*} "ls ${p#*:}/DONE_* >/dev/null 2>&1"; do sleep 60; done; done
log pools done
: > $DATA/ok_$TAG.txt
for p in $POOLS; do R ${p%%:*} "cd ${p#*:}/runs && for d in *; do ls \$d/*.npz >/dev/null 2>&1 && echo \$d; done" >> $DATA/ok_$TAG.txt; done
sort -u -o $DATA/ok_$TAG.txt $DATA/ok_$TAG.txt
awk 'NR==FNR{ok[$1]=1; next} !($1 in ok)' $DATA/ok_$TAG.txt $ITEMS > $DATA/sweep_$TAG.txt; log sweep $(grep -c . $DATA/sweep_$TAG.txt)
if [ -s $DATA/sweep_$TAG.txt ]; then W=$DATA/sweep_$TAG; NWS=${SWEEPNW:-2}; n=$(grep -c . $DATA/sweep_$TAG.txt); h=$(( (n+NWS-1)/NWS ))
  for ((w=0; w<NWS; w++)); do s=$((w*h)); e=$(( (w+1)*h )); [ $e -gt $n ] && e=$n
    [ $s -lt $e ] && (nice -n 15 $PY -m incognita_emu.runner.ens --out $W --grid $G --items $DATA/sweep_$TAG.txt --idx $s:$e >> $L 2>&1) &
  done; wait; POOLS="$POOLS local:$W"; fi
i=0; for p in $POOLS; do h=${p%%:*}; d=${p#*:}; f=ext_${TAG}_$i.npz
  if [ $h = local ]; then $PY -m incognita_emu.extract $d $DATA/$f >> $L 2>&1
  else R $h "cd ${REMOTE_REPO:-incognita-emu} && source scripts/env.sh && \$PY -m incognita_emu.extract $d $d/$f" >> $L 2>&1 && scp -q $h:$d/$f $DATA/; fi
  i=$((i+1)); done
log done; touch $DATA/$TAG.DONE
