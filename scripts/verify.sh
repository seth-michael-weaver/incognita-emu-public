#!/bin/bash
# Exact-engine verification: run an api.Emu.verify_items file with the warm ensemble runner, then
# api.Emu.compare(OUTDIR). usage: verify.sh ITEMS OUTDIR [NW=2]
source "$(dirname "$0")/env.sh" || exit 1
ITEMS=$(readlink -f $1); OUT=$(readlink -f $2); NW=${3:-2}; mkdir -p $OUT/logs
N=$(grep -c . $ITEMS); h=$(( (N+NW-1)/NW ))
for ((w=0; w<NW; w++)); do s=$((w*h)); e=$(( (w+1)*h )); [ $e -gt $N ] && e=$N
  [ $s -lt $e ] && nice -n 19 $PY -m incognita_emu.runner.ens --out $OUT --grid $G --items $ITEMS --idx $s:$e >> $OUT/logs/v$w.log 2>&1 &
done; wait; touch $OUT/VERIFY.DONE
