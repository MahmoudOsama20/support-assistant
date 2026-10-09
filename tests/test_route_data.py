from __future__ import annotations

import json
from pathlib import Path

import pytest

from route.frames import FRAMES, POOLS
from route.generate import (
    LABELS, LANGS, SPLITS, Caps, build_dataset, expand, load_massive, norm, split_frames,
    synthetic_candidates, verify, wrap,
)

SMALL = Caps(train=20, dev=5, heldout=5)
NON_OOS = ("kb_question", "data_lookup", "unsafe_request")


@pytest.fixture()
def massive(tmp_path: Path) -> Path:
    for locale, lang in (("en-US", "en"), ("ar-SA", "ar")):
        rows = []
        for part, n in (("train", 120), ("dev", 60), ("test", 30)):
            tag = "TESTPART" if part == "test" else part
            for i in range(n):
                utt = f"{tag} utterance {i} topic {i % 6}" if lang == "en" else f"{tag} جملة رقم {i} موضوع {i % 6}"
                rows.append({"partition": part, "intent": f"intent_{i % 6}", "utt": utt})
        rows.append({"partition": "train", "intent": "x", "utt": "what is my balance please" if lang == "en" else "ما هو رصيدي الآن"})
        (tmp_path / f"{locale}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp_path


def test_frames_and_pools_are_consistent() -> None:
    for label in NON_OOS:
        for lang in LANGS:
            frames = FRAMES[(label, lang)]
            assert len(frames) >= 8, (label, lang)
            for frame in frames:
                assert expand(frame, POOLS[(label, lang)]), frame  # no missing slots


def test_real_caps_are_reachable() -> None:
    caps = Caps()
    for label in NON_OOS:
        for lang in LANGS:
            for split, cap in (("train", caps.train), ("dev", caps.dev), ("heldout", caps.heldout)):
                assert len(synthetic_candidates(label, lang, split)) >= cap, (label, lang, split)


def test_split_frames_disjoint_and_nonempty() -> None:
    sp = split_frames([f"f{i} {{x}}" for i in range(10)])
    assert all(sp[s] for s in SPLITS)
    assert sum(len(v) for v in sp.values()) == 10 and len({f for v in sp.values() for f in v}) == 10


def test_wrap_lowercasing() -> None:
    assert wrap("What is it?", "Please ") == "Please what is it?"
    assert wrap("I need help", "Hi, ") == "Hi, I need help"
    assert wrap("I'm the admin", "Hi, ") == "Hi, I'm the admin"
    assert wrap("SYSTEM OVERRIDE: x", "Hi, ") == "Hi, SYSTEM OVERRIDE: x"
    assert wrap("ما هو", "من فضلك ") == "من فضلك ما هو"


def test_deterministic(massive: Path) -> None:
    a, _ = build_dataset(massive, 42, SMALL)
    b, _ = build_dataset(massive, 42, SMALL)
    assert a == b


def test_caps_and_all_cells(massive: Path) -> None:
    out, info = build_dataset(massive, 42, SMALL)
    assert not info["shortfalls"]
    caps = {"train": SMALL.train, "dev": SMALL.dev, "heldout": SMALL.heldout}
    for split in SPLITS:
        for lang in LANGS:
            for label in LABELS:
                n = sum(1 for r in out[split] if r["language"] == lang and r["label"] == label)
                assert n == caps[split], (split, lang, label, n)


def test_no_overlap_and_verify_passes(massive: Path) -> None:
    out, _ = build_dataset(massive, 42, SMALL)
    verify(out)
    texts = [{norm(r["text"]) for r in out[s]} for s in SPLITS]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])


def test_synthetic_frames_disjoint_across_splits(massive: Path) -> None:
    out, _ = build_dataset(massive, 42, SMALL)
    cells: dict[tuple[str, str, str], set[str]] = {}
    for s in SPLITS:
        for r in out[s]:
            if r["source"] == "synthetic":
                cells.setdefault((r["label"], r["language"], r["ref"]), set()).add(s)
    assert cells and all(len(v) == 1 for v in cells.values())


def test_massive_test_partition_never_used(massive: Path) -> None:
    out, _ = build_dataset(massive, 42, SMALL)
    assert not any("TESTPART" in r["text"] for s in SPLITS for r in out[s])
    with pytest.raises(ValueError):
        load_massive(massive / "en-US.jsonl", "test")


def test_support_like_massive_filtered(massive: Path) -> None:
    out, info = build_dataset(massive, 42, SMALL)
    oos = [r["text"] for s in SPLITS for r in out[s] if r["label"] == "out_of_scope"]
    assert not any("balance" in t.lower() or "رصيد" in t for t in oos)
    assert info["massive_dropped_support_like"]["en"]["train"] == 1
    assert info["massive_dropped_support_like"]["ar"]["train"] == 1