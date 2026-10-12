#!/usr/bin/env python3
"""Ensemble runner (exact engine): many (nucleus, knob point) production runs in ONE warm process, emurun semantics per
run (E2 hold rule on the run's own lines, production deck and switches, FEWRES off, PEWINDOW g, m1share r, collrule svw for
N-Z>=20), optional stage reuse (--stage-cache). Bitwise equal to emurun.py on the emulator grid.
Items lines: RUNID Z A U1,...,Uk (k = 12 or 16). Output OUT/runs/RUNID/ZZZ_AAA.npz (+ theta.json, henv.json, stagetime.json).
usage: python -m incognita_emu.runner.ens --out OUT --items F [--grid G] [--idx a:b] [--repo ENGINE] [--stage-cache DIR]"""
import argparse, json, time
from pathlib import Path
from .. import config


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--grid", default=str(config.grid54()))
    ap.add_argument("--items", required=True); ap.add_argument("--idx", default=""); ap.add_argument("--stage-cache", default="")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args(argv); out = Path(a.out).resolve(); grid = str(Path(a.grid).resolve()); items = str(Path(a.items).resolve())
    from . import prodenv as PE
    M, base = PE.boot(config.engine(a.repo))
    if a.stage_cache:
        base[config.ev("STAGE_CACHE")] = str(Path(a.stage_cache).resolve()); PE._M["base"] = base
    from physics.hf import stagetime
    from physics.hf.omp import sfunc
    DC, LOG = PE.RP.install_dir_hook(); XR = PE.RP.install_xr_hook()
    rows = [l.split() for l in open(items) if l.strip()]
    if a.idx:
        i0, i1 = (int(x) for x in a.idx.split(":")); rows = rows[i0:i1]
    for idx, r in enumerate(rows):
        rid, Z, A, us = r[0], int(r[1]), int(r[2]), [float(x) for x in r[3].split(",")]
        od = out / "runs" / rid; od.mkdir(parents=True, exist_ok=True)
        if (od / f"{Z:03d}_{A:03d}.npz").exists(): continue
        c0, w0 = time.process_time(), time.time()
        try:
            it2, (row, env) = PE.enter(Z, A, us, out / "lines", grid)
        except Exception as ex:
            (od / f"{Z:03d}_{A:03d}.err").write_text(f"h_rule: {ex!r}\n"); print(rid, "h_rule err", ex, flush=True); continue
        ch = time.process_time() - c0
        it2[0] = str(od)
        json.dump({"rid": rid, "Z": Z, "A": A, "u": us, "phys": PE.phys(us)}, open(od / "theta.json", "w"))
        json.dump({"row": row, "env": env}, open(od / "henv.json", "w"))
        LOG.clear(); DC.DIRCAP_LOG.clear(); XR.clear(); stagetime.reset(); sfunc.LAST.clear()
        st = PE.EP.run_item(M, base, it2, PE._M["repo"], "dircap,pespin", idx, {"cpu": 0.0, "wall": 0.0})
        snap = stagetime.snapshot(); snap["h_cpu_s"] = ch; snap["run_cpu_s"] = time.process_time() - c0 - ch
        json.dump(snap, open(od / "stagetime.json", "w"), default=str)
        if st == "ok":
            PE.RP.write_dir_log(DC, LOG, str(od), Z, A); PE.RP.write_xr(XR, {}, str(od), Z, A)
        else:
            (od / f"{Z:03d}_{A:03d}.err").write_text(f"run_item: {st!r}\n")
        print(rid, Z, A, st, f"h {ch:.1f}s run {snap['run_cpu_s']:.1f}s cpu, wall {time.time()-w0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
