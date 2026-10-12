#!/usr/bin/env python3
"""Per-chunk runner (one process per chunk; 12 or 16 knobs from the item's u length): production settings (E2 hold rule,
';m1share r', engine switches + S-function input, fewres off, pewindow g, collrule svw for N-Z >= 20) with a per-run lines
file = lines_P + knobs.deck(u). Bitwise equal to the warm ensemble runner (ens.py). Items: RUNID Z A U1,...,Uk [OMPPARTS].
One npz per run: OUT/runs/RUNID/ZZZ_AAA.npz (+ henv.json, theta.json, err).
usage: python -m incognita_emu.runner.emurun --out OUT --items F --idx i,j,k [--grid G] [--repo ENGINE]"""
import argparse, json, time
from pathlib import Path
from .. import config
from .. import knobs as K


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default=str(config.grid54())); ap.add_argument("--items", required=True); ap.add_argument("--idx", required=True)
    a = ap.parse_args(argv); out = Path(a.out).resolve(); repo = config.engine(a.repo)
    from . import prodenv as PE
    PP, EP, RP = PE.PP, PE.EP, PE.RP
    rows = [l.split() for l in open(a.items) if l.strip()]
    todo = [rows[int(i)] for i in a.idx.split(",")]
    M, base_env = EP._setup(repo, [], "dircap,pespin")
    from physics.hf import stagetime
    from physics.hf.omp import sfunc
    DC, LOG = RP.install_dir_hook(); XR = RP.install_xr_hook()
    for idx, r in enumerate(todo):
        rid, Z, A, us = r[0], int(r[1]), int(r[2]), [float(x) for x in r[3].split(",")]
        parts = tuple(r[4].split(",")) if len(r) > 4 else K.OMP_PARTS
        od = out / "runs" / rid; od.mkdir(parents=True, exist_ok=True)
        if (od / f"{Z:03d}_{A:03d}.npz").exists(): continue
        ld = out / "lines"; ld.mkdir(exist_ok=True); lf = ld / f"{rid}.lines"
        lf.write_text(PE.lines_text(Z, A, us, parts))
        json.dump({"rid": rid, "Z": Z, "A": A, "u": us, "phys": PE.phys(us), "omp_parts": parts}, open(od / "theta.json", "w"))
        extra = PP.DECK["P"] + (";collrule svw" if A - 2 * Z >= 20 else "")
        it = [str(od), Z, A, str(lf), a.grid, extra, f"emu:{rid}", {**PP.FLAGS, **PP.FENV}]
        t0 = time.time(); c0 = time.process_time()
        try:
            row, env = RP.h_rule(M, base_env, it)
        except Exception as ex:
            (od / f"{Z:03d}_{A:03d}.err").write_text(f"h_rule: {ex!r}\n"); print(rid, Z, A, "h_rule err", ex, flush=True); continue
        json.dump({"row": row, "env": env}, open(od / "henv.json", "w"))
        env = {**it[7], **env, config.ev("FEWRES"): "off", config.ev("PEWINDOW"): "g"}
        it2 = list(it[:5]) + [it[5] + ";m1share r", it[6], env]
        LOG.clear(); DC.DIRCAP_LOG.clear(); XR.clear(); stagetime.reset(); sfunc.LAST.clear()
        st = EP.run_item(M, base_env, it2, repo, "dircap,pespin", idx, {"cpu": 0.0, "wall": 0.0})
        if st == "ok":
            RP.write_dir_log(DC, LOG, str(od), Z, A); RP.write_xr(XR, {}, str(od), Z, A)
        else:
            (od / f"{Z:03d}_{A:03d}.err").write_text(f"run_item: {st!r}\n")
        print(rid, Z, A, st, f"{time.time()-t0:.1f}s wall {time.process_time()-c0:.1f}s cpu", flush=True)


if __name__ == "__main__":
    main()
