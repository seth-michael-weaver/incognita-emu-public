"""Put a python process into the exact production engine state for one (nucleus, knob point), without running the engine:
env (production switches + S-function input + E2 hold rule + FEWRES off + PEWINDOW g), lines file = lines_P + knobs.deck(u),
deck string = PEON (+ collrule svw if N-Z >= 20) + ';m1share r'. Same statements as emurun.main -> RP.h_rule -> EP.run_item,
in the same order. Paths: INCOGNITA_EMU_ENGINE (engine checkout), INCOGNITA_EMU_PLINES (lines_P dir)."""
import json, os
from pathlib import Path
from .prod import settings as PP
from .. import knobs as K
from .. import config
EP, RP = PP.EP, PP.RP
_M = {}


def plines():
    return config.plines()


def boot(repo=None):
    repo = repo or config.engine()
    if not _M:
        M, base = EP._setup(repo, [], "dircap,pespin")
        _M.update(M=M, base=base, repo=repo)
    return _M["M"], _M["base"]


def lines_text(Z, A, u, parts=K.OMP_PARTS):
    return K.deck(open(f"{plines()}/V32_{Z:03d}_{A:03d}.lines").read(), u, parts)


def phys(u):
    return K.phys12(u) if len(u) == 12 else K.phys(u)


def item(Z, A, u, work, grid, parts=K.OMP_PARTS, tag="emu"):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    lf = work / f"{Z:03d}_{A:03d}_{abs(hash(tuple(u))) % 10**10}.lines"
    lf.write_text(lines_text(Z, A, u, parts))
    extra = PP.DECK["P"] + (";collrule svw" if A - 2 * Z >= 20 else "")
    return [str(work), Z, A, str(lf), str(grid), extra, tag, {**PP.FLAGS, **PP.FENV}]


def enter(Z, A, u, work, grid, parts=K.OMP_PARTS, hrow=None):
    """Production state for (Z, A, u). Returns (it2, row): it2 = the run_item item, row = h-rule row (or given)."""
    M, base = boot()
    it = item(Z, A, u, work, grid, parts)
    if hrow is None:
        row, env = RP.h_rule(M, base, it)
    else:
        row, env = hrow
    env = {**it[7], **env, config.ev("FEWRES"): "off", config.ev("PEWINDOW"): "g"}
    it2 = list(it[:5]) + [it[5] + ";m1share r", it[6], env]
    fitlib, chartrun = M["fitlib"], M["chartrun"]
    os.environ.clear(); os.environ.update(base)
    for k, v in env.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = str(v)
    os.environ[fitlib.ENV_LINES_FILE] = it2[3]
    os.environ[config.ev("TALYS_FIT_LINES")] = it2[5]
    fitlib.deactivate(); fitlib._cards.cache_clear(); chartrun.drop_target_caches()
    return it2, (row, env)


def energies(grid):
    import numpy as np
    g = np.asarray(json.load(open(grid)), float)
    return tuple(float(np.float32(e)) for e in g / 1e6)
