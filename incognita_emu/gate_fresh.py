#!/usr/bin/env python3
"""Fresh-point check: per-nucleus models with the training floor 1e-3 mb, trained on all existing runs of TRAIN+VAL
nuclei, scored on fresh Sobol points 64-72 (role sob, info >= 64). Held-out nuclei: centre + axis runs only as inputs.
Also: reference = floor 1e-5 models on the same fresh points.
usage: python -m incognita_emu.gate_fresh --ext a,b,... --out DIR [--floor 1e-3] [--steps 400]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from . import emu as E


def main(argv=None):
    run(argv)


def run(argv=None):

    ap = argparse.ArgumentParser(); ap.add_argument("--ext", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=1e-3); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ref", type=float, default=1e-5); a = ap.parse_args(argv); torch.set_num_threads(1); E.configure(12)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    import pandas as pd
    D = E.load(a.ext.split(",")); PP = {k: E.prep(d) for k, d in D.items()}
    keys = [k for k in E.META["ALL"] if k in D and PP.get(k)]
    fresh = lambda r: r["role"] == "sob" and r["info"] >= 64
    old = lambda r: not fresh(r)
    for k in keys:  # fresh runs never enter anything but the FRESH test
        for r in D[k]["runs"]: r["fresh"] = fresh(r)
    AB = {}
    HN = [k for k in E.META["HN"] if k in keys]; VAL = [k for k in E.META["VAL"] if k in keys]
    TR = [k for k in keys if k not in E.META["HN"] and k not in E.META["VAL"]]
    nf = {k: sum(r["fresh"] for r in D[k]["runs"]) for k in keys}
    log = {"n_nuclei": len(keys), "fresh_runs": int(sum(nf.values())), "fresh_nuclei": int(sum(v > 0 for v in nf.values())),
           "HN_fresh_runs": int(sum(nf[k] for k in HN)), "floor": a.floor}
    print(log, flush=True); assert log["HN_fresh_runs"] == 0
    acc = E.Acc(); res = {"log": log}


    def pn_fresh(floor, setname):
        E.FLOOR = floor
        probs, own = E.problems_for(D, PP, TR + VAL, lambda r: old(r) and r["role"] in ("c", "axis", "sob", "dpil", "hpr"))
        PN = dict(zip(own, E.fit_gps(probs, a.steps)))
        for key in TR + VAL:
            te = [r for r in D[key]["runs"] if r["fresh"]]
            if not te: continue
            U = np.stack([r["u"] for r in te]); P = PP[key]
            for ch in P:
                yh = E.gp_pred(PN[(key, ch)], U)
                for i, r in enumerate(te):
                    E.FLOOR = floor; y, ok = E.Y(r, ch, P)
                    # scored cells (ok: sigma(u) > cut) have y independent of the floor
                    acc.add(setname, key, ch, r["rid"], yh[i], y, ok, P[ch]["thr"])
        print(f"{setname} done {time.time()-t0:.0f}s", flush=True)


    pn_fresh(a.ref, "FRESH_REF")      # old floor (reference, not gated)
    pn_fresh(a.floor, "FRESH")        # the fix (gated)
    E.FLOOR = a.floor
    AB = {k: E.anchors(D[k], PP[k]) for k in keys}
    assert all(AB[k] is not None for k in keys), [k for k in keys if AB[k] is None]
    # ---- PJ Jacobian (no axis runs; all old runs incl. HN nuclei's non-fresh Sobol)
    probs, own = E.problems_for(D, PP, keys, lambda r: old(r) and r["role"] in ("c", "sob", "dpil", "hpr"))
    PJ = dict(zip(own, E.fit_gps(probs, a.steps))); jrows = []; Up = np.eye(E.NK) * 0.5
    for key in keys:
        for ch in PP[key]:
            Je = AB[key][ch][0]; Jm = E.gp_pred(PJ[(key, ch)], Up) - E.gp_pred(PJ[(key, ch)], -Up)
            for k in range(E.NK):
                for c in range(Je.shape[1]): jrows.append((key, ch, k, c, float(Je[k, c]), float(Jm[k, c])))
    J = pd.DataFrame(jrows, columns=["nuc", "ch", "k", "cell", "Je", "Jm"]); J.to_parquet(out / "jacobian.parquet")
    big = J[J.Je.abs() >= 0.002]; agree = (np.sign(big.Je) == np.sign(big.Jm))
    jac = {"pooled": float(agree.mean()), "n": int(len(big)), "per_ch": {ch: float(agree[big.ch == ch].mean()) for ch in E.CHN if (big.ch == ch).any()},
           "HN": float(agree[big.nuc.isin(HN)].mean())}
    jac["pass"] = jac["pooled"] >= 0.95; print("jac", jac["pooled"], f"{time.time()-t0:.0f}s", flush=True)
    # ---- HN: A0 / A1, choice on VAL (training sets = all old sob/dpil/hpr runs)
    for setname, ks in (("VAL_A0", VAL), ("HN_A0", HN)):
        for key in ks:
            P = PP[key]; te = [r for r in D[key]["runs"] if old(r) and r["role"] in ("sob", "dpil", "hpr")]
            for ch in P:
                for r in te:
                    y, ok = E.Y(r, ch, P); acc.add(setname, key, ch, r["rid"], E.a0_pred(AB[key][ch], r["u"][None])[0], y, ok, P[ch]["thr"])


    def rows_old(ks):
        for k in ks:  # temporarily hide fresh runs from the MLP row builder
            D[k]["_all"] = D[k]["runs"]; D[k]["runs"] = [r for r in D[k]["runs"] if old(r)]
        X = E.a1_rows(D, ks, AB, PP, ("sob", "dpil", "hpr"))
        for k in ks: D[k]["runs"] = D[k]["_all"]
        return X


    Xtr, Rtr, _ = rows_old(TR); print("A1 rows", Xtr.shape, flush=True)
    nets = [E.train_mlp(Xtr, Rtr, s)[0] for s in range(5)]
    Xv, Rv, mv = rows_old(VAL); pv = E.mlp_pred(nets, Xv) + Rv[:, 2]; yv = Rv[:, 0] + Rv[:, 2]
    for i, (key, ch, rid, c) in enumerate(mv):
        acc.rows.append(("VAL_A1", key, ch, rid, bool(Rv[i, 1] > 0), bool(PP[key][ch]["thr"][c]), float(pv[i] - yv[i]), float(yv[i])))
    SV = E.summarise(acc.df(), ["VAL_A0", "VAL_A1"]); choice = {}
    for ch in E.CHN:
        s0, s1 = SV.get(f"VAL_A0:{ch}"), SV.get(f"VAL_A1:{ch}")
        if s0 is None: continue
        sc = lambda s: np.nansum([s["med"], s["p95"], s.get("thr_p95", 0) if np.isfinite(s.get("thr_p95", np.nan)) else 0])
        choice[ch] = "A1" if (s1 is not None and sc(s1) < sc(s0)) else "A0"
    print("choice", choice, flush=True)
    Xtr, Rtr, _ = rows_old(TR + VAL); nets = [E.train_mlp(Xtr, Rtr, 10 + s)[0] for s in range(5)]
    torch.save([n.state_dict() for n in nets], out / "a1_nets.pt")
    Xh, Rh, mh = rows_old(HN); ph = E.mlp_pred(nets, Xh) + Rh[:, 2]; yh = Rh[:, 0] + Rh[:, 2]
    for i, (key, ch, rid, c) in enumerate(mh):
        acc.rows.append(("HN_A1", key, ch, rid, bool(Rh[i, 1] > 0), bool(PP[key][ch]["thr"][c]), float(ph[i] - yh[i]), float(yh[i])))
    df = acc.df(); df.to_parquet(out / "cells.parquet")
    hn = df[df.set.isin(["HN_A0", "HN_A1"])].copy(); hn = hn[[f"HN_{choice.get(c, 'A0')}" == s for s, c in zip(hn.set, hn.ch)]]; hn["set"] = "HN"
    df2 = pd.concat([df, hn]); S = E.summarise(df2, ["FRESH", "FRESH_REF", "HN", "HN_A0", "HN_A1", "VAL_A0", "VAL_A1"])
    passes = {ch: bool(S.get(f"FRESH:{ch}", {}).get("pass", True) and S.get(f"HN:{ch}", {}).get("pass", True)) for ch in E.CHN}
    gate = {"PASS": bool(all(passes.values()) and jac["pass"]), "per_channel": passes, "jacobian": jac, "choice": choice,
            "summary": S, "log": log, "wall_s": time.time() - t0}
    json.dump(gate, open(out / "gate.json", "w"), indent=1, default=float)
    w = df2[df2.ok & df2.set.isin(["FRESH", "FRESH_REF", "HN"])].assign(ae=lambda x: x.err.abs()).groupby(["set", "nuc", "ch", "thr"]).ae.agg(["count", "median", lambda v: np.percentile(v, 95), "max"])
    w.columns = ["n", "med", "p95", "max"]; w.reset_index().to_csv(out / "per_nucleus.csv", index=False)
    print(json.dumps({"PASS": gate["PASS"], "per_channel": passes, "jac": jac["pooled"]}), flush=True)


if __name__ == "__main__":
    main()
