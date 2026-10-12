#!/usr/bin/env python3
"""EXACT engine sensitivity matrices from the one-at-a-time axis runs (no emulator).
S_u[k, E]   = log10 σ(u_k=+1/2) − log10 σ(u_k=−1/2)          (dex per unit design coordinate, central difference)
S_rel[k, E] = dlnσ/dln x_k  (multiplicative knobs: sA, rsc, fE1, wE1, fM1, nv1, nrv, nav, nd1, m2c)
            = dlnσ/dx_k     (additive knobs: dC [per unit ln ρ shift], dPs [per MeV])
curv[k, E]  = log10 σ(+1/2) + log10 σ(−1/2) − 2 log10 σ(0)   (second difference: non-linearity flag)
usage: python -m incognita_emu.sens --ext a.npz,b.npz,... --out DIR  -> DIR/sens.npz (float32), DIR/sens_top.csv (top-3 knobs per band)"""
import argparse, math
from pathlib import Path
import numpy as np
from . import emu as E
from . import knobs as K

BANDS = (("keV", 0, 1e6), ("1-5MeV", 1e6, 5.01e6), ("5-20MeV", 5.01e6, 2.1e7))


def dlnx_du():
    out = []
    for k in K.NAMES12:
        kind, h = K.KNOBS12[k]
        out.append(h if kind == "lin" else (math.log(h) if kind == "log" else h))  # add: dx/du = h
    return np.array(out)


def main(argv=None):
    E.configure(12); ap = argparse.ArgumentParser(); ap.add_argument("--ext", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = E.load(a.ext.split(",")); keys = [k for k in E.META["ALL"] if k in D]
    e = D[keys[0]]["e_ev"]; nN, nC, nE = len(keys), len(E.CHN), len(e)
    Su = np.full((nN, nC, E.NK, nE), np.nan, np.float32); Cv = Su.copy(); S0 = np.full((nN, nC, nE), np.nan, np.float32)
    have = np.zeros(nN, bool); g = dlnx_du(); rows = []
    for i, key in enumerate(keys):
        P = E.prep(D[key])
        if not P: continue
        ax = {(r["info"][0], r["info"][1]): r for r in D[key]["runs"] if r["role"] == "axis"}
        if len(ax) < 2 * E.NK: continue
        have[i] = True
        for j, ch in enumerate(E.CHN):
            if ch not in P: continue
            m = P[ch]["mask"]; S0[i, j, m] = P[ch]["s0"][m]
            for k in range(E.NK):
                yp, _ = E.Y(ax[(k, 1)], ch, P); ym, _ = E.Y(ax[(k, -1)], ch, P)
                Su[i, j, k, m] = yp - ym; Cv[i, j, k, m] = yp + ym
            for bn, lo, hi in BANDS:
                sel = m & (e >= lo) & (e < hi)
                if not sel.any(): continue
                mag = np.nanmedian(np.abs(Su[i, j][:, sel]), 1)
                top = np.argsort(-mag)[:3]
                rows.append({"nuc": key, "ch": ch, "band": bn, **{f"k{t+1}": K.NAMES12[top[t]] for t in range(3)},
                             **{f"s{t+1}_dex": float(mag[top[t]]) for t in range(3)}})
    Srel = Su * np.log(10.0) / g[None, None, :, None]
    np.savez_compressed(out / "sens.npz", nuclei=np.array(keys), channels=np.array(E.CHN), knobs=np.array(K.NAMES12), e_ev=e,
                        S_u=Su, S_rel=Srel.astype(np.float32), curv=Cv, sigma0=S0, have_axis=have,
                        note="S_u dex per unit u (central diff at +-1/2); S_rel dln(sigma)/dln(x) (dC: per unit ln rho, dPs: per MeV); curv second difference")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "sens_top.csv", index=False)
    print(int(have.sum()), "nuclei with exact sensitivities ->", out / "sens.npz")


if __name__ == "__main__":
    main()
