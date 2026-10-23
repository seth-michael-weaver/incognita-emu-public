"""api loads models.pt and reproduces stored EXACT engine points that were never used in training.

20 runs = verification draws vf00000-vf00003 of the 5 nuclei in data/ext/verify_unseen.npz (Fe-56, Cu-65, Au-197, Th-232,
U-238; uniform random u, run after training). Cells scored: sigma0 and sigma(u) > 1e-3 mb; threshold rows E < 1.5 E_thr
(E_thr >= 100 keV) for n2n/np/na. Limits per channel: non-threshold median <= 0.010 and p95 <= 0.030 dex; threshold rows
p95 <= 0.050 dex. The legacy models (floor 1e-5 mb) miss heavy-nucleus (n,alpha) cells near threshold (Au-197 up to 0.9 dex),
so for them test_na_threshold_rows is xfail; for the default models it is strict.
Each model set is checked against exact runs of the engine release it was trained on: default (0.4.0) ->
verify_unseen_2026-10-19.npz (engine-2026-10-19, 20 new uniform draws of the same 5 nuclei); legacy and v0_1 (the 0.1.0 default
models) -> verify_unseen.npz (engine-2026-10)."""
import numpy as np
import pytest
from conftest import REPO, models_path

EXTS = {"default": REPO / "data" / "ext" / "verify_unseen_2026-10-19.npz", "legacy": REPO / "data" / "ext" / "verify_unseen.npz",
        "v0_1": REPO / "data" / "ext" / "verify_unseen.npz"}
SETS = ["default", "legacy", "v0_1"]


@pytest.fixture(scope="module", params=SETS)
def emu_set(request):
    if not EXTS[request.param].exists(): pytest.skip(f"{EXTS[request.param].name} absent (scripts/fetch_assets.py)")
    p = models_path(request.param)
    if p is None: pytest.skip(f"models/{request.param}/models.pt absent (scripts/fetch_assets.py)")
    from incognita_emu import api
    return request.param, api.load(str(p.parent))


@pytest.fixture(scope="module")
def errors(emu_set):
    from incognita_emu import emu
    E = emu_set[1]; z = np.load(EXTS[emu_set[0]], allow_pickle=True)
    rid = [str(r) for r in z["rid"]]
    pick = [i for i, r in enumerate(rid) if r[:7] in ("vf00000", "vf00001", "vf00002", "vf00003")]
    assert len(pick) == 20
    out = {ch: {"thr": [], "nt": []} for ch in E.channels}
    for i in pick:
        key = f"{int(z['Z'][i]):03d}_{int(z['A'][i]):03d}"; md = E.models[key]
        S = E.sigma(key, z["u"][i][None])
        for ch, s in S.items():
            m = md["mask"][ch]; ex = z[emu.CH[ch]][i][m]; em = s[0][m]
            ok = ex > 1e-3
            err = np.abs(np.log10(em[ok]) - np.log10(ex[ok])); thr = md["thr"][ch][ok]
            out[ch]["thr"] += list(err[thr]); out[ch]["nt"] += list(err[~thr])
    return out


def test_models_shape(emu_set):
    E = emu_set[1]
    assert len(E.nuclei) == 185 and E.channels[:4] == ["total", "elastic", "nonel", "capture"] and len(E.e_ev) == 54 and E.nk == 12


@pytest.mark.parametrize("ch", ["total", "elastic", "nonel", "capture", "inel", "n2n", "np", "na", "fission"])
def test_non_threshold_within_hp_error(errors, ch):
    e = np.array(errors[ch]["nt"])
    if not len(e): pytest.skip("no cells")
    assert np.median(e) <= 0.010 and np.percentile(e, 95) <= 0.030, (np.median(e), np.percentile(e, 95))


@pytest.mark.parametrize("ch", ["n2n", "np"])
def test_threshold_rows(errors, ch):
    e = np.array(errors[ch]["thr"])
    if not len(e): pytest.skip("no cells")
    assert np.percentile(e, 95) <= 0.050, np.percentile(e, 95)


def test_na_threshold_rows(emu_set, errors, request):
    if emu_set[0] == "legacy":
        request.applymarker(pytest.mark.xfail(reason="legacy models: heavy-Z (n,alpha) threshold rows (Au-197 up to 0.9 dex)", strict=False))
    e = np.array(errors["na"]["thr"])
    print(emu_set[0], "na thr n", len(e), "p95", np.percentile(e, 95), "max", e.max())
    assert np.percentile(e, 95) <= 0.050, np.percentile(e, 95)


def test_jacobian_and_tmc_run(emu_set):
    E = emu_set[1]
    J = E.jacobian("026_056", np.zeros(12), ["capture"])
    assert J["capture"].shape == (12, 54) and np.isfinite(J["capture"][:, E.models["026_056"]["mask"]["capture"]]).all()
    T = E.tmc("026_056", n=20000, channels=["capture"])
    assert np.nanmin(T["capture"]["sd"]) >= 0
