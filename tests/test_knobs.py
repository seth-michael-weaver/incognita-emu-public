"""knobs -> deck lines byte-identical to the decks behind the shipped models, on 50 random u per knob set (golden sha256 in
fixtures)."""
import hashlib, json
import numpy as np
import pytest
from conftest import FIX
from incognita_emu import knobs as K

G = json.load(open(FIX / "knobs_golden.json"))
KEYS = ["020_040", "026_056", "092_238"]
h = lambda s: hashlib.sha256(s.encode()).hexdigest()


@pytest.mark.parametrize("key", KEYS)
def test_deck12_golden(key):
    base = (FIX / "lines" / f"V32_{key}.lines").read_text()
    assert [h(K.deck12(base, u)) for u in G["U12"]] == G["deck12"][key]
    assert [h(K.deck(base, u)) for u in G["U12"]] == G["deck12"][key]       # 16-knob deck with 12 entries == deck12


@pytest.mark.parametrize("key", KEYS)
def test_deck16_golden(key):
    base = (FIX / "lines" / f"V32_{key}.lines").read_text()
    assert [h(K.deck(base, u)) for u in G["U16"]] == G["deck16"][key]


def test_deck16_zero_extras_equals_deck12():
    base = (FIX / "lines" / "V32_092_238.lines").read_text()
    for u in G["U12"][:10]:
        assert K.deck(base, list(u) + [0.0] * 4) == K.deck12(base, u)


def test_phys12_golden():
    for u, p in zip(G["U12"], G["phys12"]):
        assert K.phys12(u) == p
