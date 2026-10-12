"""design generators reproduce the item files behind the shipped models byte for byte."""
import hashlib, json
import pytest
from incognita_emu import design as D

md5 = lambda rows: hashlib.md5(("\n".join(rows) + "\n").encode()).hexdigest()


def test_pilot(): assert md5(D.pilot_items()) == D.ITEM_MD5["pilot/items_all.txt"]


def test_stage1():
    r1, r2, m = D.stage1_items()
    assert md5(r1) == D.ITEM_MD5["stage1/items_r1.txt"] and md5(r2) == D.ITEM_MD5["stage1/items_r2.txt"]
    M = D.meta("design_meta_stage1.json")
    assert all(M[k] == m[k] for k in m)
    assert len(m["ALL"]) == 185 and len(m["HN"]) == 19 and len(m["VAL"]) == 18


def test_2b():
    q, al = D.items_2b()
    assert md5(q) == D.ITEM_MD5["2b/items_2b.txt"] and md5(al) == D.ITEM_MD5["2b/items_2b_all.txt"]


def test_fresh(): assert md5(D.fresh_items()) == D.ITEM_MD5["fresh/items_na.txt"]


def test_nucleus_layout():
    rows = D.nucleus_items("026_056", nsob=32)
    assert len(rows) == 1 + 24 + 32 and rows[0].startswith("c_026_056 26 56 ")
    _, r2, _ = D.stage1_items()
    # a new nucleus's Sobol points are the stage-1 points (same seed rule): points 32..63 of a stage-1 nucleus
    k = r2[0].split()[0][5:]
    assert D.nucleus_items(k, nsob=32, sob0=32)[25:] == [r for r in r2 if r.split()[0].endswith(k)]
    rows16 = D.nucleus_items("092_238", nsob=16, nk=16)
    assert len(rows16) == 1 + 32 + 16
