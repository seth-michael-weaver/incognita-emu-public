#!/usr/bin/env python3
"""Per-nucleus emulators (PN GP), Jacobian check (PJ), held-out-nucleus models (A0 separable anchors, A1 residual MLP) and
accuracy scoring. configure(nk=12|16, floor=...) selects the 12- or 16-knob variant. Data = extract npz files
(incognita_emu.extract).

usage:
  python -m incognita_emu.emu gate  --ext a.npz,b.npz,... --out DIR [--nk 12|16] [--floor 1e-5] [--steps 400]
  python -m incognita_emu.emu final --ext ... --out DIR [--nk 12|16] [--floor ...]   # PN on ALL runs -> DIR/models.pt
Roles by run id: c_KEY centre; a<kk><p|m>_KEY / j<kk><p|m>_KEY axis (u_k = +-0.5); s<jjj>_KEY Sobol (HP iff j%5==4);
t<jjj>_KEY 16-knob Sobol (HP2B iff j%5==4); d<iiii> space-filling design (HP iff rank within nucleus %5==4);
v<cc>_<nnnnn>_KEY random verification points (HPR, never trained by `gate`).

Training floor: y = log10 max(sigma(u), FLOOR) - log10 sigma0. Default FLOOR = CUT = 1e-3 mb on all channels (models/default).
The legacy models used FLOOR = 1e-5 mb (--floor 1e-5). FLOOR may be a float (all channels) or a {channel: floor} dict."""
import argparse, json, os, re, time
from pathlib import Path
import numpy as np
import torch

from . import gp as T
from .config import META as METADIR

CH = {"total": "x:tot_total", "elastic": "x:tot_elastic", "nonel": "x:tot_nonelastic", "capture": "x:capture_mb",
      "inel": "x:ch_xs100000", "n2n": "x:ch_xs200000", "np": "x:ch_xs010000", "na": "x:ch_xs000001",
      "fission": "x:tot_fission"}
CHN = list(CH)
THRCH = ("n2n", "np", "na")
CUT = 1e-3
FLOOR = CUT             # float or {channel: floor}; default 1e-3 mb (legacy models: 1e-5)
NK = 12
FISS = (14, 15)         # fvc, fbc: active only for Z >= 90 (16-knob variant)
DEV, F64 = T.DEV, T.F64
META = json.load(open(METADIR / "design_meta_stage1.json"))
if os.environ.get("INCOGNITA_EMU_SPLIT"): META.update(json.load(open(os.environ["INCOGNITA_EMU_SPLIT"])))  # smoke tests only


def configure(nk=None, floor=None):
    global NK, FLOOR
    if nk is not None:
        assert nk in (12, 16), nk
        NK = nk
    if floor is not None: FLOOR = floor


def floor_of(ch):
    return FLOOR.get(ch, 1e-5) if isinstance(FLOOR, dict) else FLOOR


def sob_roles():
    return ("sob", "dpil", "sob2") if NK == 16 else ("sob", "dpil")


def role(rid):
    m = re.match(r"^c_(\d{3}_\d{3})$", rid)
    if m: return "c", None
    m = re.match(r"^[aj](\d{2})([pm])_(\d{3}_\d{3})$", rid)
    if m:
        k = int(m.group(1))
        if k >= NK: return None, None
        return "axis", (k, 1 if m.group(2) == "p" else -1)
    m = re.match(r"^s(\d{3})_(\d{3}_\d{3})$", rid)
    if m: return "sob", int(m.group(1))
    if NK == 16:
        m = re.match(r"^t(\d{3})_(\d{3}_\d{3})$", rid)
        if m: return "sob2", int(m.group(1))
    m = re.match(r"^d(\d{4})$", rid)
    if m: return "dpil", int(m.group(1))
    m = re.match(r"^v(\d{2})_(\d{5})_(\d{3}_\d{3})$", rid)          # random verification draws (uniform prior)
    if m: return "hpr", int(m.group(2))
    return None, None


def load(paths):
    """{key: {"e_ev", "runs": [ {rid, role, info, u, ch: sigma(54)} ]}}; later files override the same (key, rid)."""
    D = {}
    for p in paths:
        z = np.load(p, allow_pickle=True); files = set(z.files)
        X = {c: (z[k] if k in files else None) for c, k in CH.items()}
        rid, Z, A, U, e = z["rid"], z["Z"], z["A"], z["u"], z["e_ev"]
        for i in range(len(rid)):
            r, info = role(str(rid[i]))
            if r is None: continue
            key = f"{int(Z[i]):03d}_{int(A[i]):03d}"
            d = D.setdefault(key, {"e_ev": e, "runs": {}})
            u = U[i].astype(float)
            if NK == 16: u = np.pad(u, (0, NK - len(u)))
            d["runs"][str(rid[i])] = {"rid": str(rid[i]), "role": r, "info": info, "u": u,
                                      **{c: (X[c][i].astype(float) if X[c] is not None else np.zeros(len(e))) for c in CHN}}
    for key, d in D.items():
        runs = list(d["runs"].values())
        dp = sorted([r for r in runs if r["role"] == "dpil"], key=lambda r: r["info"])
        for k, r in enumerate(dp): r["hp"] = (k % 5 == 4)
        for r in runs:
            if r["role"] in ("sob", "sob2"): r["hp"] = (r["info"] % 5 == 4)
            elif r["role"] != "dpil": r["hp"] = False
        d["runs"] = runs
    return D


def prep(d):
    """per channel: centre, mask (sigma0 > CUT), thr flag per masked cell, E_thr"""
    c = [r for r in d["runs"] if r["role"] == "c"]
    if not c: return None
    c = c[0]; e = d["e_ev"]; out = {}
    for ch in CHN:
        s0 = c[ch]; m = s0 > CUT
        if m.sum() == 0: continue
        ethr = float(e[m][0]) if ch in THRCH and e[m][0] >= 1e5 else 0.0
        thr = (e[m] < 1.5 * ethr) if ethr > 0 else np.zeros(int(m.sum()), bool)
        out[ch] = {"s0": s0, "mask": m, "thr": thr, "ethr": ethr}
    return out


def Y(run, ch, P):
    m, s0 = P[ch]["mask"], P[ch]["s0"][P[ch]["mask"]]
    s = run[ch][m]
    return np.log10(np.maximum(s, floor_of(ch))) - np.log10(s0), s > CUT


def active(key):
    return [k for k in range(NK) if k not in FISS or int(key[:3]) >= 90]


def anchors(d, P, key=None):
    """A0 coefficients per channel: a (NK, cells), b (NK, cells), zero rows for inactive knobs; None if an active axis
    run is missing"""
    ax = {(r["info"][0], r["info"][1]): r for r in d["runs"] if r["role"] == "axis"}
    act = active(key) if (NK == 16 and key is not None) else list(range(NK))
    if any((k, s) not in ax for k in act for s in (1, -1)): return None
    out = {}
    for ch in P:
        n = int(P[ch]["mask"].sum()); yp = np.zeros((NK, n)); ym = np.zeros((NK, n))
        for k in act: yp[k] = Y(ax[(k, 1)], ch, P)[0]; ym[k] = Y(ax[(k, -1)], ch, P)[0]
        out[ch] = (yp - ym, 2.0 * (yp + ym))
    return out


def a0_pred(ab, U):
    a, b = ab
    return U @ a + (U ** 2) @ b


def fit_gps(problems, steps, chunk=192):
    out = []
    for i in range(0, len(problems), chunk):
        out += T.fit_gp_batch(problems[i:i + chunk], steps=steps)
        if DEV == "cuda": torch.cuda.empty_cache()
    return [m for m, _ in out]


def gp_pred(m, U):
    with torch.no_grad():
        return T.pred_gp(m, torch.as_tensor(np.atleast_2d(U), dtype=F64, device=DEV)).cpu().numpy()


# ---------------- scoring ----------------
class Acc:
    def __init__(self): self.rows = []

    def add(self, set_, key, ch, rid, yhat, y, ok, thr, ethr_e=None):
        for c in range(len(y)):
            self.rows.append((set_, key, ch, rid, bool(ok[c]), bool(thr[c]), float(yhat[c] - y[c]), float(y[c])))

    def df(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=["set", "nuc", "ch", "rid", "ok", "thr", "err", "y"])


# accuracy limits (E1) used by `gate`
E1 = {"med": 0.010, "p95": 0.030, "thr_p95": 0.050, "jac_sign": 0.95}


def summarise(df, sets):
    G = {}
    for s in sets:
        for ch in CHN:
            q = df[(df.set == s) & (df.ch == ch)]
            if not len(q): continue
            ok = q[q.ok]; nt, th = ok[~ok.thr], ok[ok.thr]
            e, et = nt.err.abs().values, th.err.abs().values
            r = {"cells": int(len(nt)), "thr_cells": int(len(th)), "closed": int((~q.ok).sum()), "nuclei": int(q.nuc.nunique()),
                 "med": float(np.median(e)) if len(e) else np.nan, "p95": float(np.percentile(e, 95)) if len(e) else np.nan,
                 "thr_p95": float(np.percentile(et, 95)) if len(et) else np.nan,
                 "thr_med": float(np.median(et)) if len(et) else np.nan,
                 "zero_med": float(np.median(nt.y.abs())) if len(nt) else np.nan,
                 "zero_p95": float(np.percentile(nt.y.abs(), 95)) if len(nt) else np.nan}
            r["pass"] = bool((r["med"] <= E1["med"] or not len(e)) and (r["p95"] <= E1["p95"] or not len(e))
                             and (not len(et) or r["thr_p95"] <= E1["thr_p95"]))
            # nucleus bootstrap CI of median / p95
            if s.startswith("HN") and len(e):
                rng = np.random.default_rng(1); nucs = nt.nuc.unique(); g = {n: nt[nt.nuc == n].err.abs().values for n in nucs}
                bs = []
                for _ in range(500):
                    pick = rng.choice(nucs, len(nucs)); v = np.concatenate([g[n] for n in pick]); bs.append((np.median(v), np.percentile(v, 95)))
                bs = np.array(bs); r["med_ci"] = list(np.percentile(bs[:, 0], [2.5, 97.5])); r["p95_ci"] = list(np.percentile(bs[:, 1], [2.5, 97.5]))
            G[f"{s}:{ch}"] = r
    return G


# ---------------- A1 residual MLP ----------------
def a1_rows(D, keys, AB, PP, use_sets=None):
    """feature rows for the residual MLP: per (run, channel, cell)"""
    use_sets = use_sets or sob_roles()
    X, R, meta = [], [], []
    chi = {c: i for i, c in enumerate(CHN)}
    for key in keys:
        d, P, ab = D[key], PP[key], AB[key]
        e = d["e_ev"]
        for r in d["runs"]:
            if r["role"] not in use_sets: continue
            for ch in P:
                y, ok = Y(r, ch, P); m = P[ch]["mask"]; ee = e[m]
                a, b = ab[ch]; base = r["u"] @ a + (r["u"] ** 2) @ b
                et = P[ch]["ethr"]; tc = np.clip((ee - et) / et, -1, 5) if et > 0 else np.full(len(ee), 5.0)
                oh = np.zeros((len(ee), len(CHN))); oh[:, chi[ch]] = 1
                f = np.concatenate([np.repeat(r["u"][None], len(ee), 0), a.T, b.T, np.log10(ee)[:, None] - 5,
                                    tc[:, None], np.log10(P[ch]["s0"][m])[:, None] / 3, oh], 1)
                X.append(f); R.append(np.c_[y - base, ok, base]); meta += [(key, ch, r["rid"], c) for c in range(len(ee))]
    return np.concatenate(X).astype(np.float32), np.concatenate(R).astype(np.float32), meta


def train_mlp(X, R, seed, epochs=60, bs=4096, lr=2e-3, hid=256):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ok = R[:, 1] > 0; X, y = X[ok], R[ok, 0]
    n = len(X); idx = rng.permutation(n); nv = n // 10; iv, it = idx[:nv], idx[nv:]
    Xt = torch.as_tensor(X, device=DEV); yt = torch.as_tensor(y, device=DEV)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hid), torch.nn.SiLU(), torch.nn.Linear(hid, hid), torch.nn.SiLU(),
                              torch.nn.Linear(hid, hid), torch.nn.SiLU(), torch.nn.Linear(hid, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, bstate, bad = 1e9, None, 0
    itt = torch.as_tensor(it, device=DEV); ivt = torch.as_tensor(iv, device=DEV)
    for ep in range(epochs):
        net.train(); perm = itt[torch.randperm(len(itt), device=DEV)]
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]; opt.zero_grad()
            loss = torch.nn.functional.huber_loss(net(Xt[j])[:, 0], yt[j], delta=0.01); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            v = torch.cat([(net(Xt[ivt[i:i + 65536]])[:, 0] - yt[ivt[i:i + 65536]]).abs() for i in range(0, len(ivt), 65536)]).median().item()
        if v < best - 1e-6: best, bstate, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    net.load_state_dict(bstate); net.eval()
    return net, best


def mlp_pred(nets, X):
    Xt = torch.as_tensor(X, device=DEV); out = []
    with torch.no_grad():
        for i in range(0, len(Xt), 65536):
            out.append(torch.stack([n(Xt[i:i + 65536])[:, 0] for n in nets]).mean(0).cpu().numpy())
    return np.concatenate(out)


# ---------------- commands ----------------
def problems_for(D, PP, keys, include):
    probs, own = [], []
    for key in keys:
        d, P = D[key], PP[key]
        tr = [r for r in d["runs"] if include(r)]
        U = np.stack([r["u"] for r in tr])
        for ch in P:
            Yt = np.stack([Y(r, ch, P)[0] for r in tr])
            probs.append((U, Yt)); own.append((key, ch))
    return probs, own


def cmd_gate(a):
    import pandas as pd
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    SR = sob_roles(); b16 = NK == 16
    D = load(a.ext.split(",")); PP = {k: prep(d) for k, d in D.items()}
    keys = [k for k in META["ALL"] if k in D and PP.get(k)]
    missing = [k for k in META["ALL"] if k not in keys]
    AB = {k: anchors(D[k], PP[k], k) for k in keys}
    noax = [k for k in keys if AB[k] is None]
    HN = [k for k in META["HN"] if k in keys and AB[k] is not None]; VAL = [k for k in META["VAL"] if k in keys and AB[k] is not None]
    TR = [k for k in keys if k not in META["HN"] and k not in META["VAL"] and AB[k] is not None]
    log = {"nk": NK, "floor": FLOOR, "n_nuclei": len(keys), "missing": missing, "no_axis": noax, "HN": HN, "VAL": VAL, "n_train": len(TR),
           "runs": {k: {r: sum(1 for x in D[k]["runs"] if x["role"] == r) for r in ("c", "axis", "hpr") + SR} for k in keys}}
    print(len(keys), "nuclei;", len(noax), "without full axis;", "HN", len(HN), "VAL", len(VAL), "TR", len(TR), flush=True)
    acc = Acc()
    # ---- PN: per-nucleus GP on centre + axis + non-HP Sobol of TR+VAL nuclei; test HP and HPR
    pn_keys = TR + VAL
    probs, own = problems_for(D, PP, pn_keys, lambda r: r["role"] in ("c", "axis") or (r["role"] in SR and not r["hp"]))
    print("PN problems", len(probs), flush=True)
    PN = dict(zip(own, fit_gps(probs, a.steps))); print(f"PN fit {time.time()-t0:.0f}s", flush=True)
    hp_sets = ((("HP2B", lambda r: r["role"] == "sob2" and r["hp"]), ("HP12", lambda r: r["role"] in ("sob", "dpil") and r["hp"]))
               if b16 else (("HP", lambda r: r["role"] in ("sob", "dpil") and r["hp"]),)) + (("HPR", lambda r: r["role"] == "hpr"),)
    for key in pn_keys:
        d, P = D[key], PP[key]
        for setname, sel in hp_sets:
            te = [r for r in d["runs"] if sel(r)]
            if not te: continue
            U = np.stack([r["u"] for r in te])
            for ch in P:
                yh = gp_pred(PN[(key, ch)], U)
                for i, r in enumerate(te):
                    y, ok = Y(r, ch, P); acc.add(setname, key, ch, r["rid"], yh[i], y, ok, P[ch]["thr"])
    # ---- PJ: no axis runs; Jacobian vs engine axis differences
    probs, own = problems_for(D, PP, keys, lambda r: r["role"] == "c" or (r["role"] in SR and (not r["hp"])))
    PJ = dict(zip(own, fit_gps(probs, a.steps))); print(f"PJ fit {time.time()-t0:.0f}s", flush=True)
    jrows = []
    Up = np.eye(NK) * 0.5
    for key in keys:
        if AB[key] is None: continue
        for ch in PP[key]:
            Je = AB[key][ch][0]                               # y(+1/2) - y(-1/2), (NK, cells)
            Jm = gp_pred(PJ[(key, ch)], Up) - gp_pred(PJ[(key, ch)], -Up)
            for k in (active(key) if b16 else range(NK)):
                for c in range(Je.shape[1]):
                    jrows.append((key, ch, k, c, float(Je[k, c]), float(Jm[k, c])))
    J = pd.DataFrame(jrows, columns=["nuc", "ch", "k", "cell", "Je", "Jm"]); J.to_parquet(out / "jacobian.parquet")
    big = J[J.Je.abs() >= 0.002]; agree = (np.sign(big.Je) == np.sign(big.Jm))
    jac = {"pooled": float(agree.mean()), "n": int(len(big)),
           "per_ch": {ch: float(agree[big.ch == ch].mean()) for ch in CHN if (big.ch == ch).any()},
           "trio": float(agree[big.nuc.isin(META["TRIO"])].mean()) if big.nuc.isin(META["TRIO"]).any() else None,
           "HN": float(agree[big.nuc.isin(HN)].mean()),
           "corr_trio": {f"{n}:{ch}": float(np.corrcoef(g.Je, g.Jm)[0, 1]) for (n, ch), g in J[J.nuc.isin(META["TRIO"])].groupby(["nuc", "ch"]) if g.Je.std() > 0}}
    if b16:
        jac.update({"per_knob": {int(k): float(agree[big.k == k].mean()) for k in range(NK) if (big.k == k).any()},
                    "n_per_knob": {int(k): int((big.k == k).sum()) for k in range(NK)},
                    "new_knobs": float(agree[big.k >= 12].mean()) if (big.k >= 12).any() else None})
    jac["pass"] = jac["pooled"] >= E1["jac_sign"]
    # ---- HN: A0 and A1 (choice on VAL)
    te_roles = SR + ("hpr",)
    for setname, ks in (("VAL_A0", VAL), ("HN_A0", HN)):
        for key in ks:
            d, P = D[key], PP[key]
            te = [r for r in d["runs"] if r["role"] in te_roles]
            for ch in P:
                for r in te:
                    y, ok = Y(r, ch, P); acc.add(setname, key, ch, r["rid"], a0_pred(AB[key][ch], r["u"][None])[0], y, ok, P[ch]["thr"])
    Xtr, Rtr, _ = a1_rows(D, TR, AB, PP)
    print("A1 rows", Xtr.shape, flush=True)
    nets = [train_mlp(Xtr, Rtr, s)[0] for s in range(5)]; print(f"A1 trained {time.time()-t0:.0f}s", flush=True)
    Xv, Rv, mv = a1_rows(D, VAL, AB, PP, te_roles)
    pv = mlp_pred(nets, Xv) + Rv[:, 2]
    yv = Rv[:, 0] + Rv[:, 2]
    for i, (key, ch, rid, c) in enumerate(mv):
        acc.rows.append(("VAL_A1", key, ch, rid, bool(Rv[i, 1] > 0), bool(PP[key][ch]["thr"][c]), float(pv[i] - yv[i]), float(yv[i])))
    dfv = acc.df(); SV = summarise(dfv, ["VAL_A0", "VAL_A1"])
    choice = {}
    for ch in CHN:
        s0, s1 = SV.get(f"VAL_A0:{ch}"), SV.get(f"VAL_A1:{ch}")
        if s0 is None: continue
        sc = lambda s: np.nansum([s["med"], s["p95"], s.get("thr_p95", 0) if np.isfinite(s.get("thr_p95", np.nan)) else 0])
        choice[ch] = "A1" if (s1 is not None and sc(s1) < sc(s0)) else "A0"
    print("HN choice", choice, flush=True)
    json.dump({"choice": choice, "VAL": SV}, open(out / "hn_choice.json", "w"), indent=1, default=float)
    # refit A1 on TR+VAL, score HN
    Xtr, Rtr, _ = a1_rows(D, TR + VAL, AB, PP)
    nets = [train_mlp(Xtr, Rtr, 10 + s)[0] for s in range(5)]
    torch.save([n.state_dict() for n in nets], out / "a1_nets.pt")
    Xh, Rh, mh = a1_rows(D, HN, AB, PP, te_roles)
    ph = mlp_pred(nets, Xh) + Rh[:, 2]; yh = Rh[:, 0] + Rh[:, 2]
    for i, (key, ch, rid, c) in enumerate(mh):
        acc.rows.append(("HN_A1", key, ch, rid, bool(Rh[i, 1] > 0), bool(PP[key][ch]["thr"][c]), float(ph[i] - yh[i]), float(yh[i])))
    df = acc.df(); df.to_parquet(out / "cells.parquet")
    hn = df[df.set.isin(["HN_A0", "HN_A1"])].copy()
    hn = hn[[f"HN_{choice.get(c, 'A0')}" == s for s, c in zip(hn.set, hn.ch)]]
    if b16:   # HN2B = 2b Sobol points (gated), HN12 = 12-knob points (reported)
        is2b = hn.rid.str.match(r"^t\d{3}_")
        hn2, hn1 = hn[is2b].copy(), hn[~is2b].copy(); hn2["set"] = "HN2B"; hn1["set"] = "HN12"
        df2 = pd.concat([df, hn2, hn1]); gsets = ("HP2B", "HN2B")
        S = summarise(df2, ["HP2B", "HP12", "HPR", "HN2B", "HN12", "HN_A0", "HN_A1", "VAL_A0", "VAL_A1"])
    else:
        hn["set"] = "HN"; df2 = pd.concat([df, hn]); gsets = ("HP", "HN")
        S = summarise(df2, ["HP", "HPR", "HN", "HN_A0", "HN_A1", "VAL_A0", "VAL_A1"])
    passes = {ch: bool(all(S.get(f"{g}:{ch}", {}).get("pass", True) for g in gsets)) for ch in CHN}
    gate = {"E1": bool(all(passes.values()) and jac["pass"]), "per_channel": passes, "jacobian": jac, "choice": choice,
            "summary": S, "log": log, "wall_s": time.time() - t0}
    json.dump(gate, open(out / "gate.json", "w"), indent=1, default=float)
    wsets = ["HP2B", "HN2B", "HP12", "HN12"] if b16 else ["HP", "HN"]
    w = df2[df2.ok & df2.set.isin(wsets)].assign(ae=lambda x: x.err.abs()).groupby(["set", "nuc", "ch"]).ae.agg(["median", lambda v: np.percentile(v, 95)])
    w.columns = ["med", "p95"]; w.reset_index().to_csv(out / "per_nucleus.csv", index=False)
    print(json.dumps({"E1": gate["E1"], "per_channel": passes, "jac": jac["pooled"]}), flush=True)


def cmd_final(a):
    from .knobs import names
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = load(a.ext.split(",")); PP = {k: prep(d) for k, d in D.items()}
    keys = [k for k in META["ALL"] if k in D and PP.get(k)]
    probs, own = problems_for(D, PP, keys, lambda r: r["role"] in ("c", "axis", "hpr") + sob_roles())
    ms = fit_gps(probs, a.steps)
    models = {}
    for (key, ch), m in zip(own, ms):
        P = PP[key][ch]
        md = models.setdefault(key, {"centre": {}, "mask": {}, "thr": {}, "m": {}, "n": len(probs[own.index((key, ch))][0])})
        md["centre"][ch] = P["s0"]; md["mask"][ch] = P["mask"]; md["thr"][ch] = P["thr"]
        md["m"][ch] = {k: (v.cpu() if torch.is_tensor(v) else ([t.cpu() for t in v] if isinstance(v, list) else v)) for k, v in m.items()}
    e_ev = D[keys[0]]["e_ev"]
    torch.save({"models": models, "e_ev": e_ev, "channels": CHN, "knobs": list(names(NK)), "floor": FLOOR, "cut": CUT},
               out / "models.pt")
    print(len(models), "nuclei saved", out / "models.pt")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gate", "final"]); ap.add_argument("--ext", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--nk", type=int, default=12); ap.add_argument("--floor", type=float, default=None)
    a = ap.parse_args(argv); torch.set_num_threads(1)
    configure(a.nk, a.floor)
    {"gate": cmd_gate, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
