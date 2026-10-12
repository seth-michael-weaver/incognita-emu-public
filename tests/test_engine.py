"""Exact-engine smoke test (marker `engine`, opt-in: pytest -m engine): the package runner reproduces the stored
reference centre run of Fe-56 (fixtures/c_026_056) bitwise on every channel array. Needs scripts/env.sh variables."""
import os, subprocess
from pathlib import Path
import numpy as np
import pytest
from conftest import FIX, REPO

need = ["INCOGNITA_EMU_ENGINE", "INCOGNITA_EMU_TALYS_DIR"]
pytestmark = [pytest.mark.engine, pytest.mark.skipif(any(not os.environ.get(v) for v in need), reason="engine env not set")]


def test_centre_run_bitwise(tmp_path):
    items = tmp_path / "items.txt"; items.write_text("c_026_056 26 56 " + ",".join(["0.000000"] * 12) + "\n")
    out = Path(os.environ.get("INCOGNITA_EMU_TEST_OUT", tmp_path / "out"))
    cmd = f"source {REPO}/scripts/env.sh && $PY -m incognita_emu.runner.ens --out {out} --items {items}"
    subprocess.run(["bash", "-c", cmd], check=True, cwd=REPO)
    a = np.load(out / "runs/c_026_056/026_056.npz", allow_pickle=True); b = np.load(FIX / "c_026_056/026_056.npz", allow_pickle=True)
    keys = [k for k in b.files if k.startswith(("tot_", "ch_xs", "capture_mb"))]
    assert keys
    for k in keys: np.testing.assert_array_equal(a[k], b[k], err_msg=k)
