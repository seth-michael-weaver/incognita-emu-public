#!/usr/bin/env python3
"""Per-item collective-E2 hold rule, computed in the same warm worker just before the run:
  h = 1 - (G(B+E) - G(B)) / G_dip(B+E),  B = item deck incl. ';m1share r',  E = switch ISOE2=e,  G_dip = G(B+E) - G(B+E, hold 0)
  run env = {ISOE2: e, ISOHOLD: 'Zc,Ac,h'}   (switch names carry the engine prefix, engine_pool.ev)
  If G_dip <= 0 (no collective-E2 strength to hold) h is undefined: run with ISOE2=e and no hold (G(B+E) == G(B) then), tag hrule=nodip.
Items: [out_dir, Z, A, lines_file, grid_json, extra_lines(WITHOUT m1share r), tag, {}].
Also: read-only hooks that keep the direct-reaction sums and the channel extras of every run (dir_/xr_ZZZ_AAA.json[.gz])."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_pool as EP  # noqa: E402
ev = EP.ev

KEYS = tuple(ev(k) for k in ("ISOE2", "SPINQUENCH", "ISOHOLD", "ISOE2_K", "M1SHARE"))


def h_rule(M, base_env, it):
    np, fitlib, chartrun, warmrun = M["np"], M["fitlib"], M["chartrun"], M["warmrun"]
    _, Z, A, lf, gj, extra, tag = it[:7]
    os.environ.clear(); os.environ.update(base_env)
    for k, v in (it[7] if len(it) > 7 and it[7] else {}).items():   # item's own env (None = unset)
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = str(v)
    os.environ[fitlib.ENV_LINES_FILE] = lf
    os.environ[ev("TALYS_FIT_LINES")] = (extra or "") + ";m1share r"
    e_mev = (float(np.float32(json.load(open(gj))[0] / 1e6)),)
    from physics.hf.ingredients import ingredients

    def ing(env):
        for k in KEYS: os.environ.pop(k, None)
        os.environ.update(env)
        fitlib.deactivate(); fitlib._cards.cache_clear(); chartrun.drop_target_caches()
        cas = warmrun.new_cascade(Z, A, e_mev)
        d = ingredients(cas, None)
        r = (float(d["ing_d0theo_ev"]), float(d["ing_gamgam_th_ev"]), int(d["ing_Zc"]), int(d["ing_Ac"]))
        cas = None; chartrun.drop_target_caches(); return r

    t0 = time.process_time(); row = dict(target=f"{Z:03d}_{A:03d}", Z=Z, A=A)
    d0, g0, zc, ac = ing({}); row.update(Zc=zc, Ac=ac, d0_R=d0, gg_R=g0)
    d_on, g_on, *_ = ing({ev("ISOE2"): "e"})
    _, g_z, *_ = ing({ev("ISOE2"): "e", ev("ISOHOLD"): f"{zc},{ac},0"})
    g_dip = g_on - g_z
    h = 1.0 - (g_on - g0) / g_dip if g_dip > 0 else float("nan")
    row.update(gg_RE=g_on, gdip_R=g_dip, h_R=h)
    if np.isfinite(h):
        env = {ev("ISOE2"): "e", ev("ISOHOLD"): f"{zc},{ac},{h!r}"}; row["hrule"] = "hold"
    else:
        env = {ev("ISOE2"): "e"}; row["hrule"] = "nodip"
    for k in KEYS: os.environ.pop(k, None)
    row["h_cpu_s"] = time.process_time() - t0
    return row, env


def install_dir_hook():
    """Read-only wrapper on physics.hf.direct.chain.direct_result (same record fields), so every production run
    carries its own rule-D direct-sum log dir_ZZZ_AAA.json (no separate DIR reruns)."""
    import physics.hf.direct.chain as DC
    from physics.hf.direct import dwbaunit
    orig = DC.direct_result; LOG = []

    def wrap(target, e_inc_mev, energies, **kw):
        res, fg = orig(target, e_inc_mev, energies, **kw)
        try:
            d = float(res.xsdirdisctot_mb) + float(res.xscollconttot_mb) + float(res.xsgrcoll_mb.sum())
            LOG.append(dict(target=str(target), e_mev=float(e_inc_mev), xsdirdisctot=float(res.xsdirdisctot_mb), xscollconttot=float(res.xscollconttot_mb),
                            xsgrcoll=float(res.xsgrcoll_mb.sum()), xsgrtot=float(res.xsgrtot_mb), direct=d, evaluating=bool(dwbaunit.evaluating())))
        except Exception as exc:
            LOG.append(dict(err=f"{type(exc).__name__}: {exc}"))
        return res, fg
    DC.direct_result = wrap
    return DC, LOG


def install_xr_hook():
    """Read-only capture: per incident energy the ChannelResult extras the C path computes but
    the npz does not keep (isomeric splits per channel, residual isomer branches, gamma/fission channel parts, particle production,
    multiplicities, exclusive totals) + stagetime per-stage CPU/wall. Written to xr_ZZZ_AAA.json.gz; never feeds the curves."""
    from physics.hf.emission import channels as CHM
    orig = CHM._channel_result; XR = []

    def d1(x): return {str(k): float(v) for k, v in (x or {}).items()}

    def d2(x): return {str(k): d1(v) for k, v in (x or {}).items()}

    def wrap(*a, **k):
        r = orig(*a, **k)
        try:
            XR.append({"iso": d2(r.xschaniso_mb), "ch": d1(r.xschannel_mb), "gam": d1(r.xsgamchannel_mb), "fis": d1(r.xsfischannel_mb),
                       "excl": d1(r.xsexclusive_mb), "exclcont": d1(r.xsexclcont_mb), "part": d1(r.xsparticle_mb), "mult": d1(r.multiplicity),
                       "fistot": float(r.xsfistot_mb), "br": {f"{z}-{a_}": d1(v) for (z, a_), v in r.xsbranch.items()},
                       "rp": {f"{z}-{a_}": float(v) for (z, a_), v in r.residual_mb.items()}})
        except Exception as exc:
            XR.append({"err": f"{type(exc).__name__}: {exc}"})
        return r
    CHM._channel_result = wrap
    return XR


def write_xr(XR, st, out, Z, A):
    import gzip
    with gzip.open(Path(out) / f"xr_{Z:03d}_{A:03d}.json.gz", "wt") as fh: json.dump({"Z": Z, "A": A, "per_energy": XR, "stagetime": st}, fh)


def write_dir_log(DC, LOG, out, Z, A):
    from physics.hf.preeq.chain import incident_reaction_xs
    try:
        es = sorted({r["e_mev"] for r in LOG if "e_mev" in r}); sr = incident_reaction_xs(Z, A, tuple(es)) if es else {}
        for r in LOG:
            if "e_mev" in r: r["sigR_sph"] = float(sr[round(r["e_mev"], 6)])
    except Exception as exc:
        LOG.append(dict(err_sr=f"{type(exc).__name__}: {exc}"))
    json.dump(dict(Z=Z, A=A, log=LOG, dircap_log=[list(map(float, t[2:])) for t in DC.DIRCAP_LOG if (t[0], t[1]) == (Z, A)]),
              open(Path(out) / f"dir_{Z:03d}_{A:03d}.json", "w"))
    DC.DIRCAP_LOG.clear()


