import importlib, re
import pytest

MODS = ["incognita_emu", "incognita_emu.config", "incognita_emu.knobs", "incognita_emu.design", "incognita_emu.gp",
        "incognita_emu.emu", "incognita_emu.gate_fresh", "incognita_emu.extract", "incognita_emu.sens", "incognita_emu.mf33",
        "incognita_emu.api", "incognita_emu.runner.prod.settings", "incognita_emu.runner.prodenv", "incognita_emu.runner.ens",
        "incognita_emu.runner.emurun"]


@pytest.mark.parametrize("m", MODS)
def test_import(m):
    importlib.import_module(m)


def test_no_hardcoded_home():
    from pathlib import Path
    pkg = Path(__file__).resolve().parent.parent / "incognita_emu"
    for f in pkg.rglob("*.py"):
        s = f.read_text()
        assert not re.search(r"/(home|Users)/|~/", s), f
