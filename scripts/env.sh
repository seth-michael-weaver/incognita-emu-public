# Production engine environment for exact runs. Source it; it exports PY, G (grid54) and REPO.
# Required: INCOGNITA_EMU_ENGINE (engine checkout with .venv), INCOGNITA_EMU_TALYS_DIR (scripts/make_structure.sh).
# Optional: INCOGNITA_EMU_RIPL_LEVELS (RIPL-4 levels-param.data; default $INCOGNITA_EMU_TALYS_DIR/ripl4/levels-param.data),
#           INCOGNITA_EMU_PLINES (default data/lines_P), INCOGNITA_EMU_CC_CACHE (default cache/cc; speed only),
#           INCOGNITA_EMU_SLTABLE (S-function table; default incognita_emu/meta/sfunc_ripl.csv, 'off' = off), INCOGNITA_EMU_PY,
#           INCOGNITA_EMU_ENGINE_PREFIX (prefix of the engine's switches, default INCOGNITA).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for v in INCOGNITA_EMU_ENGINE INCOGNITA_EMU_TALYS_DIR; do
  [ -n "${!v}" ] || { echo "env.sh: $v is not set (README.md 'Configuration')" >&2; return 1 2>/dev/null || exit 1; }
done
P=${INCOGNITA_EMU_ENGINE_PREFIX:-INCOGNITA}
export INCOGNITA_EMU_PLINES=${INCOGNITA_EMU_PLINES:-$REPO/data/lines_P} INCOGNITA_EMU_CC_CACHE=${INCOGNITA_EMU_CC_CACHE:-$REPO/cache/cc}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LC_ALL=C UV_NO_SYNC=1
for v in $(env | grep -o "^${P}_[A-Z0-9_]*" | grep -v '^INCOGNITA_EMU_'); do unset $v; done
export TALYS_DIR=$INCOGNITA_EMU_TALYS_DIR ${P}_RIPL_LEVELS_PARAM=${INCOGNITA_EMU_RIPL_LEVELS:-$INCOGNITA_EMU_TALYS_DIR/ripl4/levels-param.data}
export ${P}_GGDEADEND=1 ${P}_NGNEXCL=1 ${P}_DWBAUNIT=1 ${P}_LEVCOMPLETE=tgt
export ${P}_BANDCHECK=1 ${P}_TRAPFIX=1 ${P}_WFCNORM=1
export ${P}_CC_CACHE=$INCOGNITA_EMU_CC_CACHE ${P}_TALYS_FIT=nodata ${P}_ENGPHYS=dircap,pespin ${P}_SPECOUT_DA=0
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
ENG=$INCOGNITA_EMU_ENGINE; PY=${INCOGNITA_EMU_PY:-$ENG/.venv/bin/python}; G=$REPO/incognita_emu/meta/grid54.json
mkdir -p "$INCOGNITA_EMU_CC_CACHE"
