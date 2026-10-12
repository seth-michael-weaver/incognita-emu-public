#!/usr/bin/env python3
"""MF33-style relative covariance demo (knob-prior propagation; NOT calibrated: parameter spread under-covers real
uncertainty).
For each demo nucleus and channel: emulator TMC (uniform prior on the 12 knobs, N draws, GPU) -> relative covariance on
the grid54 points; check against the ENGINE's own prior sample (every Sobol/design/random run of the nucleus = draws from
the same uniform prior): relative sd ratio and correlation-matrix agreement. Writes an ENDF-6 MF33 LB=5 (LS=1, symmetric)
section per (nucleus, MT) with group boundaries at geometric midpoints of the grid points.
usage: python -m incognita_emu.mf33 --models DIR --ext a.npz,... --out DIR [--nuclei 026_056,...] [--n 1000000]"""
import argparse
from pathlib import Path
import numpy as np
from . import emu as E
from . import api as EA

MT = {"total": 1, "elastic": 2, "nonel": 3, "capture": 102, "inel": 4, "n2n": 16, "np": 103, "na": 107, "fission": 18}
DEMO = ["026_056", "040_090", "079_197", "082_208", "092_238", "029_065", "090_232"]


def fnum(x):
    """ENDF 11-column float"""
    if x == 0: return " 0.000000+0"
    s = f"{x:.6e}"; m, e = s.split("e"); e = int(e)
    return f"{m}{'+' if e >= 0 else '-'}{abs(e)}".rjust(11)


def rec(fields, mat, mf, mt, ns):
    line = "".join(f if isinstance(f, str) else fnum(f) if isinstance(f, float) else f"{f:11d}" for f in fields)
    return f"{line:<66}{mat:4d}{mf:2d}{mt:3d}{ns:5d}"


def mf33_section(za, awr, mat, mt, ebounds, cov):
    """one MF33 section: HEAD, subsection CONT (MT1 = MT, NC=0, NI=1), LIST LB=5 LS=1 (upper triangle incl. diagonal)"""
    ne = len(ebounds); k = ne - 1
    tri = [cov[i, j] for i in range(k) for j in range(i, k)]
    vals = list(ebounds) + tri; lines = []; ns = 1
    lines.append(rec([float(za), float(awr), 0, 0, 0, 1], mat, 33, mt, ns)); ns += 1
    lines.append(rec([0.0, 0.0, 0, mt, 0, 1], mat, 33, mt, ns)); ns += 1
    lines.append(rec([0.0, 0.0, 1, 5, len(vals), ne], mat, 33, mt, ns)); ns += 1
    for i in range(0, len(vals), 6):
        lines.append(rec([float(v) for v in vals[i:i + 6]] + [""] * (6 - len(vals[i:i + 6])), mat, 33, mt, ns)); ns += 1
    lines.append(f"{'':66}{mat:4d}{33:2d}{0:3d}{99999:5d}")
    return lines


def main(argv=None):
    E.configure(12); ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--nuclei", default=",".join(DEMO)); ap.add_argument("--n", type=int, default=1_000_000)
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    EM = EA.load(a.models); D = E.load(a.ext.split(",")); e = EM.e_ev
    eb = np.concatenate([[e[0] / np.sqrt(e[1] / e[0])], np.sqrt(e[1:] * e[:-1]), [e[-1] * np.sqrt(e[-1] / e[-2])]])
    rows, endf = [], []
    for key in a.nuclei.split(","):
        if key not in EM.models or key not in D: print("skip", key); continue
        T = EM.tmc(key, n=a.n, seed=7)
        P = E.prep(D[key]); smp = [r for r in D[key]["runs"] if r["role"] in ("sob", "dpil", "hpr")]
        z, A = int(key[:3]), int(key[4:])
        for ch, t in T.items():
            m = EM.models[key]["mask"][ch]
            Sx = np.stack([r[ch][m] for r in smp]); mu = Sx.mean(0)
            Cx = np.cov(Sx.T) / np.outer(mu, mu)
            Ce = t["relcov"][np.ix_(m, m)]
            rsd_e, rsd_x = np.sqrt(np.diag(Ce)), np.sqrt(np.clip(np.diag(Cx), 0, None))
            live = (rsd_x > 1e-4)
            Re = Ce / np.outer(rsd_e, rsd_e); Rx = Cx / np.outer(np.maximum(rsd_x, 1e-30), np.maximum(rsd_x, 1e-30))
            iu = np.triu_indices(int(live.sum()), 1)
            dR = np.abs(Re[np.ix_(live, live)] - Rx[np.ix_(live, live)])[iu] if live.sum() > 1 else np.array([np.nan])
            rows.append({"nuc": key, "ch": ch, "n_engine": len(smp), "cells": int(m.sum()),
                         "rsd_emu_med": float(np.median(rsd_e)), "rsd_engine_med": float(np.median(rsd_x)),
                         "rsd_ratio_med": float(np.median(rsd_e[live] / rsd_x[live])) if live.any() else np.nan,
                         "corr_absdiff_med": float(np.nanmedian(dR)), "corr_absdiff_p95": float(np.nanpercentile(dR, 95))})
            # MF33 on the emulated cells; zero rows/cols off the modelled cells
            C = np.zeros((len(e), len(e))); C[np.ix_(m, m)] = Ce
            endf += mf33_section(1000 * z + A, float(A), 100 * z + 25, MT[ch], eb, C)
            np.savez_compressed(out / f"relcov_{key}_{ch}.npz", e_ev=e, ebounds=eb, relcov_emu=t["relcov"], relcov_engine_cells=Cx,
                                mask=m, mean=t["mean"], sd=t["sd"], q=t["q"], n_tmc=a.n, n_engine=len(smp))
    (out / "MF33_demo.endf").write_text("\n".join(endf) + "\n")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "mf33_check.csv", index=False); print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
