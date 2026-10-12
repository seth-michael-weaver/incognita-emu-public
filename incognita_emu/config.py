"""All machine-specific locations come from environment variables (or CLI flags that override them). Nothing in the
package hard-codes a home directory. Unset variables raise a clear error only when a code path needs them.

  INCOGNITA_EMU_MODELS    directory holding models.pt (default: <repo>/models/default; legacy: models/legacy)
  INCOGNITA_EMU_DATA      directory holding extract npz files and verification runs (default: <repo>/data)
  INCOGNITA_EMU_ENGINE    engine checkout with its .venv (repository incognita, tag ENGINE_TAG; scripts/get_engine.sh)
  INCOGNITA_EMU_PLINES    directory of the engine input files V32_ZZZ_AAA.lines (default: <repo>/data/lines_P)
  INCOGNITA_EMU_CC_CACHE  coupled-channels cache directory; speed only, results are identical (default: <repo>/cache/cc)
  INCOGNITA_EMU_TALYS_DIR TALYS structure directory, as built by scripts/make_structure.sh
  INCOGNITA_EMU_RIPL_LEVELS RIPL-4 levels-param.data (default: $INCOGNITA_EMU_TALYS_DIR/ripl4/levels-param.data)
  INCOGNITA_EMU_SLTABLE   measured S-function table, csv Z,A,S0,S1 (default meta/sfunc_ripl.csv; 'off' = input off)
  INCOGNITA_EMU_PY        python of the engine venv (default: $INCOGNITA_EMU_ENGINE/.venv/bin/python)
  INCOGNITA_EMU_ENGINE_PREFIX  prefix of the engine's own environment switches (default INCOGNITA)"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
META = PKG / "meta"
ENGINE_TAG = "engine-2026-10"       # public engine release the shipped models were trained on
ENGINE_PREFIX = os.environ.get("INCOGNITA_EMU_ENGINE_PREFIX", "INCOGNITA")


def ev(name):
    """name of the engine environment switch `name` (e.g. ev("FEWRES") -> INCOGNITA_FEWRES)"""
    return f"{ENGINE_PREFIX}_{name}"


def get(name, default=None, required=False):
    v = os.environ.get(name)
    if v: return os.path.expanduser(v)
    if default is not None: return str(default)
    if required:
        raise RuntimeError(f"{name} is not set (see incognita_emu/config.py and README.md 'Configuration')")
    return None


def models_dir(override=None): return Path(override or get("INCOGNITA_EMU_MODELS", REPO / "models" / "default"))
def data_dir(override=None): return Path(override or get("INCOGNITA_EMU_DATA", REPO / "data"))
def engine(override=None): return override or get("INCOGNITA_EMU_ENGINE", required=True)
def plines(override=None): return override or get("INCOGNITA_EMU_PLINES", REPO / "data" / "lines_P")
def sltable(): return get("INCOGNITA_EMU_SLTABLE", META / "sfunc_ripl.csv")
def cc_cache(): return get("INCOGNITA_EMU_CC_CACHE", REPO / "cache" / "cc")
def grid54(): return META / "grid54.json"
