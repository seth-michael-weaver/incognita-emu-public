#!/usr/bin/env python3
"""GP / quadratic per-nucleus emulators + GPU Total Monte Carlo / Saltelli. fit_gp_batch takes the knob count from the
problems (U.shape[1]), so 12- and 16-knob fits can share one process.

Per (nucleus, channel): y = log10 max(sigma(u), FLOOR) - log10 sigma(0) on the cells with sigma(0) > 1e-3 mb.
  Q  : ridge on [1, u, u_i u_j (i<=j)], lambda by exact LOO over a grid (intercept unpenalised);
  GP : k = b + sum_k w_k u_k u'_k + sf^2 exp(-0.5 sum_k ((u_k-u'_k)/l_k)^2) + sn^2 delta, hyper-parameters shared
       across the channel's cells (columns standardised), type-II ML by Adam on log hyper-parameters; exact LOO via K^-1.
Campaign CLI (12 knobs, 6 channels): python -m incognita_emu.gp fit|run|verify ... (see cmd_* below)."""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

CH = {"capture": "x:capture_mb", "n2n": "x:ch_xs200000", "np": "x:ch_xs010000", "na": "x:ch_xs000001",
      "inel": "x:ch_xs100000", "total": "x:tot_total"}
CHN = list(CH)
CUT, FLOOR = 1e-3, 1e-5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
F64 = torch.float64
NK = 12
IU = np.triu_indices(NK)


def feats(U):
    """[1, u, u_i u_j (i<=j)] for U (n, 12) torch."""
    q = (U[:, :, None] * U[:, None, :])[:, IU[0], IU[1]]
    return torch.cat([torch.ones(U.shape[0], 1, dtype=U.dtype, device=U.device), U, q], 1)


def load(paths):
    D = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        z = {k: z[k] for k in ["rid", "Z", "A", "u", "e_ev"] + [v for v in CH.values() if v in z.files]}   # npz re-reads per access
        z["files"] = list(z)
        for i in range(len(z["rid"])):
            key = f"{int(z['Z'][i]):03d}_{int(z['A'][i]):03d}"
            r = {"rid": str(z["rid"][i]), "u": z["u"][i].astype(float)}
            for c, k in CH.items():
                r[c] = z[k][i].astype(float) if k in z["files"] else np.zeros(len(z["e_ev"]))
            D.setdefault(key, {})[r["rid"]] = r          # later files override (same rid = same run)
        e_ev = z["e_ev"]
    return D, e_ev


def nucleus_data(runs, key):
    c = runs.get(f"c_{key}")
    if c is None: return None
    rows = [r for rid, r in sorted(runs.items())]
    U = np.stack([r["u"] for r in rows])
    out = {"U": U, "rid": [r["rid"] for r in rows], "centre": {}, "Y": {}, "mask": {}}
    for ch in CHN:
        s0 = c[ch]; m = s0 > CUT
        S = np.stack([r[ch] for r in rows])
        out["centre"][ch] = s0; out["mask"][ch] = m
        out["Y"][ch] = np.log10(np.maximum(S[:, m], FLOOR)) - np.log10(s0[m])[None, :]
    return out


# ---------------- Q: ridge quadratic ----------------
LAMS = 10.0 ** np.arange(-10, 3.01, 0.5)


def fit_q(U, Y):
    Ut = torch.as_tensor(U, dtype=F64, device=DEV); Yt = torch.as_tensor(Y, dtype=F64, device=DEV)
    P = feats(Ut); n, p = P.shape
    Dg = torch.ones(p, dtype=F64, device=DEV); Dg[0] = 0.0
    best = None
    for lam in LAMS:
        G = P.T @ P + lam * torch.diag(Dg) + 1e-14 * torch.eye(p, dtype=F64, device=DEV)
        Gi = torch.linalg.inv(G)
        H = P @ Gi @ P.T; beta = Gi @ (P.T @ Yt)
        R = (Yt - P @ beta) / (1 - torch.diagonal(H))[:, None].clamp(min=1e-12)
        mse = float((R ** 2).mean())
        if best is None or mse < best[0]: best = (mse, lam, beta, R)
    return {"kind": "Q", "lam": best[1], "beta": best[2]}, best[3].cpu().numpy()


def pred_q(m, U):
    return feats(U) @ m["beta"]


# ---------------- GP ----------------
def _K(th, U1, U2, same):
    b, w, sf, ell, sn = th
    K = b + (U1 * w) @ U2.T
    d2 = (((U1[:, None, :] - U2[None, :, :]) / ell) ** 2).sum(-1)
    K = K + sf ** 2 * torch.exp(-0.5 * d2)
    if same: K = K + (sn ** 2 + 1e-10) * torch.eye(U1.shape[0], dtype=F64, device=DEV)
    return K


def _unpack(z):
    return (torch.exp(z[0]), torch.exp(z[1:NK + 1]), torch.exp(z[NK + 1]), torch.exp(z[NK + 2:2 * NK + 2]), torch.exp(z[2 * NK + 2]))


def fit_gp_batch(problems, steps=400, lr=0.05):
    """All (nucleus, channel) GPs at once on the GPU: problems = list of (U (n,12), Y (n,m)). Padded rows get an
    identity kernel block and zero targets (adds exactly 0 to the objective); padded columns are zero. Adam on log
    hyper-parameters, objective = sum_b NLL_b / (n_b m_b). Returns models and exact-LOO residuals per problem."""
    NK = problems[0][0].shape[1]
    B = len(problems); nmax = max(p[0].shape[0] for p in problems); mmax = max(max(p[1].shape[1], 1) for p in problems)
    U = torch.zeros(B, nmax, NK, dtype=F64, device=DEV); Z = torch.zeros(B, nmax, mmax, dtype=F64, device=DEV)
    rm = torch.zeros(B, nmax, dtype=torch.bool, device=DEV); nm = torch.zeros(B, dtype=F64, device=DEV)
    stats = []
    for b, (u, y) in enumerate(problems):
        n, m = y.shape
        yt = torch.as_tensor(y, dtype=F64, device=DEV)
        mu = yt.mean(0); sd = yt.std(0).clamp(min=1e-12); live = yt.std(0) > 1e-10
        z = ((yt - mu) / sd)[:, live]
        U[b, :n] = torch.as_tensor(u, dtype=F64, device=DEV); Z[b, :n, :z.shape[1]] = z; rm[b, :n] = True
        nm[b] = n * max(z.shape[1], 1)
        stats.append((mu, sd, live, n, z.shape[1]))
    mcols = torch.tensor([max(s[4], 0) for s in stats], dtype=F64, device=DEV)
    z0 = torch.tensor([np.log(0.1)] + [np.log(1.0 / NK)] * NK + [np.log(0.3)] + [np.log(1.5)] * NK + [np.log(1e-3)],
                      dtype=F64, device=DEV).repeat(B, 1).requires_grad_(True)
    eye = torch.eye(nmax, dtype=F64, device=DEV)
    pad = (~rm).to(F64)
    rr = (rm[:, :, None] & rm[:, None, :]).to(F64)

    def kern(z):
        b, w, sf, ell, sn = (torch.exp(z[:, 0]), torch.exp(z[:, 1:NK + 1]), torch.exp(z[:, NK + 1]), torch.exp(z[:, NK + 2:2 * NK + 2]), torch.exp(z[:, 2 * NK + 2]))
        K = b[:, None, None] + torch.einsum("bik,bjk->bij", U * w[:, None, :], U)
        d2 = ((((U[:, :, None, :] - U[:, None, :, :]) / ell[:, None, None, :]) ** 2).sum(-1))
        K = K + sf[:, None, None] ** 2 * torch.exp(-0.5 * d2)
        K = K * rr + eye * ((sn ** 2 + 1e-10)[:, None, None] * rm[:, :, None].to(F64) + pad[:, :, None])
        return K
    opt = torch.optim.Adam([z0], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        K = kern(z0); L = torch.linalg.cholesky(K)
        A = torch.cholesky_solve(Z, L)
        nll = 0.5 * (Z * A).sum((1, 2)) + mcols * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        loss = (nll / nm).sum() + 1e-4 * (z0[:, NK + 2:2 * NK + 2] ** 2).sum()
        loss.backward(); opt.step()
        with torch.no_grad(): z0[:, 2 * NK + 2].clamp_(min=np.log(1e-6))
    out = []
    with torch.no_grad():
        K = kern(z0); Ki = torch.linalg.inv(K)
        for b, (mu, sd, live, n, m) in enumerate(stats):
            th = (torch.exp(z0[b, 0]), torch.exp(z0[b, 1:NK + 1]), torch.exp(z0[b, NK + 1]), torch.exp(z0[b, NK + 2:2 * NK + 2]), torch.exp(z0[b, 2 * NK + 2]))
            Kib = Ki[b, :n, :n]; A = Kib @ Z[b, :n, :m]
            loo = torch.zeros(n, len(mu), dtype=F64, device=DEV)
            loo[:, live] = (A / torch.diagonal(Kib)[:, None]) * sd[live]
            out.append(({"kind": "GP", "th": list(th), "U": U[b, :n].clone(), "A": A, "mu": mu, "sd": sd, "live": live},
                        loo.cpu().numpy()))
    return out


def pred_gp(m, U, chunk=8192):
    out = []
    for i in range(0, U.shape[0], chunk):
        Us = U[i:i + chunk]
        Ks = _K(m["th"], Us, m["U"], False)
        y = m["mu"].expand(Us.shape[0], -1).clone()
        y[:, m["live"]] = y[:, m["live"]] + (Ks @ m["A"]) * m["sd"][m["live"]]
        out.append(y)
    return torch.cat(out, 0)


def predict(m, U):
    return pred_q(m, U) if m["kind"] == "Q" else pred_gp(m, U)


# ---------------- commands ----------------
def cmd_fit(a):
    runs, e_ev = load(a.ext.split(","))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hold = set(a.exclude.split(",")) if a.exclude else set()
    models, rows = {}, []
    t0 = time.time()
    probs, owners, nds = [], [], {}
    for key in sorted(runs):
        nd = nucleus_data({r: v for r, v in runs[key].items() if r not in hold}, key)
        if nd is None: print("no centre", key); continue
        nds[key] = nd
        models[key] = {"centre": nd["centre"], "mask": nd["mask"], "n": len(nd["rid"]), "m": {}}
        for ch in CHN:
            Y = nd["Y"][ch]
            if Y.shape[1] == 0: continue
            mq, lq = fit_q(nd["U"], Y)
            models[key]["m"][ch] = {"Q": mq}
            probs.append((nd["U"], Y)); owners.append((key, ch, lq))
    print("Q done", len(probs), f"{time.time()-t0:.0f}s", flush=True)
    res = fit_gp_batch(probs, steps=a.steps)
    print("GP done", f"{time.time()-t0:.0f}s", flush=True)
    for (key, ch, lq), (mg, lg) in zip(owners, res):
        models[key]["m"][ch]["GP"] = mg
        Y = nds[key]["Y"][ch]
        for kind, L in (("Q", lq), ("GP", lg)):
            e = np.abs(L).ravel()
            rows.append({"nuc": key, "ch": ch, "model": kind, "n": len(nds[key]["rid"]), "cells": Y.shape[1],
                         "loo_med": float(np.median(e)), "loo_p95": float(np.percentile(e, 95)),
                         "ysd_med": float(np.median(Y.std(0)))})
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(out / "loo.csv", index=False)
    # model choice per channel: pooled median of per-(nucleus) LOO medians weighted by cells -> simply pooled cells
    choice = {}
    for ch in CHN:
        s = df[df.ch == ch].groupby("model").apply(lambda g: np.average(g.loo_med, weights=g.cells))
        choice[ch] = str(s.idxmin()) if len(s) else "GP"
    json.dump({"choice": choice, "pooled": {ch: df[df.ch == ch].groupby("model").loo_med.median().to_dict() for ch in CHN},
               "exclude": sorted(hold), "e_ev": [float(x) for x in e_ev]}, open(out / "choice.json", "w"), indent=1)
    torch.save({"models": models, "choice": choice, "e_ev": e_ev}, out / "models.pt")
    print(json.dumps(choice))


def _stats(S):
    """S (N, m) sigma draws -> mean, sd, quantiles (5, m), relative covariance (m, m)"""
    mu = S.mean(0); sd = S.std(0)
    q = torch.quantile(S, torch.tensor([0.025, 0.16, 0.5, 0.84, 0.975], dtype=S.dtype, device=S.device), dim=0)
    X = (S - mu) / mu
    C = X.T @ X / (S.shape[0] - 1)
    return mu, sd, q, C


def cmd_run(a):
    out = Path(a.out); M = torch.load(out / "models.pt", weights_only=False)
    models, choice, e_ev = M["models"], M["choice"], M["e_ev"]
    dz = np.load(a.design); NUC = list(dz["NUC"]); U = dz["U"]; SA, SB = dz["SA"], dz["SB"]
    nE = len(e_ev); nN = len(NUC)
    R = {k: np.full((nN, len(CHN)) + s, np.nan, np.float32) for k, s in
         (("mean", (nE,)), ("sd", (nE,)), ("q", (5, nE)), ("relcov", (nE, nE)), ("S1", (NK, nE)), ("ST", (NK, nE)),
          ("lmean", (nE,)), ("lsd", (nE,)))}
    R["xcorr_cap_n2n"] = np.full((nN, nE, nE), np.nan, np.float32)
    n_tmc = n_sal = 0; t_tmc = t_sal = 0.0
    torch.cuda.synchronize() if DEV == "cuda" else None
    T0 = time.time()
    for i, key in enumerate(NUC):
        if key not in models: print("skip", key); continue
        md = models[key]
        Ut = torch.as_tensor(U[i], dtype=F64, device=DEV)
        A = torch.as_tensor(SA, dtype=F64, device=DEV); B = torch.as_tensor(SB, dtype=F64, device=DEV)
        ABs = []
        for k in range(NK):
            X = A.clone(); X[:, k] = B[:, k]; ABs.append(X)
        Usal = torch.cat([A, B] + ABs, 0)
        lsig = {}
        for j, ch in enumerate(CHN):
            if ch not in md["m"]: continue
            m = md["m"][ch][choice[ch]]; mask = md["mask"][ch]
            s0 = torch.as_tensor(md["centre"][ch][mask], dtype=F64, device=DEV)
            t1 = time.time()
            Yd = predict(m, Ut)                                          # (17000, cells) log10 ratio
            if DEV == "cuda": torch.cuda.synchronize()
            t_tmc += time.time() - t1
            S = s0 * 10.0 ** Yd
            mu, sd, q, C = _stats(S)
            idx = np.flatnonzero(mask)
            R["mean"][i, j, idx] = mu.cpu().numpy(); R["sd"][i, j, idx] = sd.cpu().numpy()
            R["q"][i, j][:, idx] = q.cpu().numpy(); R["relcov"][i, j][np.ix_(idx, idx)] = C.cpu().numpy()
            R["lmean"][i, j, idx] = Yd.mean(0).cpu().numpy(); R["lsd"][i, j, idx] = Yd.std(0).cpu().numpy()
            lsig[ch] = (idx, S)
            # Saltelli / Jansen on y = log10 sigma
            t1 = time.time()
            F = predict(m, Usal)
            if DEV == "cuda": torch.cuda.synchronize()
            t_sal += time.time() - t1
            N = A.shape[0]; fA, fB = F[:N], F[N:2 * N]
            V = torch.cat([fA, fB], 0).var(0).clamp(min=1e-30)
            for k in range(NK):
                fABk = F[(2 + k) * N:(3 + k) * N]
                R["S1"][i, j, k, idx] = ((fB * (fABk - fA)).mean(0) / V).cpu().numpy()
                R["ST"][i, j, k, idx] = (0.5 * ((fA - fABk) ** 2).mean(0) / V).cpu().numpy()
        if "capture" in lsig and "n2n" in lsig:
            (ia, Sa), (ib, Sb) = lsig["capture"], lsig["n2n"]
            Xa = (Sa - Sa.mean(0)) / Sa.std(0).clamp(min=1e-300); Xb = (Sb - Sb.mean(0)) / Sb.std(0).clamp(min=1e-300)
            R["xcorr_cap_n2n"][i][np.ix_(ia, ib)] = (Xa.T @ Xb / (Xa.shape[0] - 1)).cpu().numpy()
        n_tmc += U.shape[1]; n_sal += Usal.shape[0]
        print(key, f"{time.time()-T0:.1f}s", flush=True)
    wall = time.time() - T0
    np.savez_compressed(out / "campaign.npz", NUC=np.array(NUC), CH=np.array(CHN), e_ev=e_ev, **R)
    tim = {"device": DEV, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "wall_s_total": wall,
           "tmc_evals": n_tmc, "saltelli_evals": n_sal, "evals_total": n_tmc + n_sal,
           "predict_s_tmc_all_channels": t_tmc, "predict_s_saltelli_all_channels": t_sal,
           "evals_per_s_predict_only": (n_tmc + n_sal) / max(t_tmc + t_sal, 1e-9),
           "evals_per_s_end_to_end": (n_tmc + n_sal) / wall,
           "note": "one evaluation = one (nucleus, knob vector) -> 6 channels x 54 energies"}
    json.dump(tim, open(out / "timing.json", "w"), indent=1); print(json.dumps(tim, indent=1))


def cmd_verify(a):
    out = Path(a.out); M = torch.load(out / "models.pt", weights_only=False)
    models, choice, e_ev = M["models"], M["choice"], M["e_ev"]
    cz = np.load(out / "campaign.npz"); NUC = list(cz["NUC"])
    dz = np.load(a.design); U = dz["U"]
    items = {l.split()[0]: l.split() for l in open(a.items) if l.strip()}
    rows = []
    for d in sorted(Path(a.runs, "runs").iterdir()):
        f = list(d.glob("*.npz"))
        if not f or d.name not in items: continue
        z = np.load(f[0], allow_pickle=True)
        it = items[d.name]; key = d.name.split("_", 2)[2]; n_i = NUC.index(key); draw = int(d.name.split("_")[1])
        u = np.array([float(x) for x in it[3].split(",")])
        md = models.get(key)
        if md is None: continue
        for j, ch in enumerate(CHN):
            if ch not in md["m"]: continue
            k = CH[ch][2:]; s = z[k].astype(float) if k in z.files else np.zeros(len(e_ev))
            mask = md["mask"][ch]; s0 = md["centre"][ch][mask]
            for kind in ("Q", "GP"):
                yhat = predict(md["m"][ch][kind], torch.as_tensor(u[None], dtype=F64, device=DEV))[0].cpu().numpy()
                yex = np.log10(np.maximum(s[mask], FLOOR)) - np.log10(s0)
                ok = s[mask] > CUT
                ee = e_ev[mask]
                # threshold rows: first modelled energy of the channel's centre curve
                ethr = ee[0] if ch in ("n2n", "np", "na") and len(ee) else 0.0
                thr = (ee < 1.5 * ethr) if ethr > 0 and ch == "n2n" else np.zeros_like(ok)
                # PIT vs this nucleus's emulated TMC distribution (chosen model only): from stored quantiles
                q = cz["q"][n_i, j][:, mask]
                for c in np.flatnonzero(ok):
                    rows.append({"rid": d.name, "nuc": key, "A": int(key[4:]), "ch": ch, "model": kind, "chosen": kind == choice[ch],
                                 "e_ev": float(ee[c]), "thr": bool(thr[c]), "err": float(yhat[c] - yex[c]), "y": float(yex[c]),
                                 "in68": bool(q[1, c] <= s[mask][c] <= q[3, c]), "in95": bool(q[0, c] <= s[mask][c] <= q[4, c]),
                                 "tmc_lsd": float(cz["lsd"][n_i, j][mask][c])})
            nlow = int(((s[mask] <= CUT)).sum())
            if nlow: rows.append({"rid": d.name, "nuc": key, "ch": ch, "model": "closed", "n_closed": nlow})
    import pandas as pd
    df = pd.DataFrame(rows); df.to_parquet(out / "verify_cells.parquet")
    print(len(set(df.rid)), "verification runs read")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--ext"); ap.add_argument("--out", required=True)
    ap.add_argument("--design"); ap.add_argument("--runs"); ap.add_argument("--items"); ap.add_argument("--exclude", default=""); ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(1)
    {"fit": cmd_fit, "run": cmd_run, "verify": cmd_verify}[a.cmd](a)
