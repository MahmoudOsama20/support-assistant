import random
import string
from collections import Counter

from route.frames import FRAMES, POOLS
from route.generate import _bin, pick_length_matched


def test_pick_length_matched_follows_target() -> None:
    rows = [{"utt": " ".join(["w"] * k) + f" {i}", "intent": f"i{i % 3}"}
            for i in range(60) for k in (1, 3, 5, 8, 12)]
    target = [2] * 5 + [9] * 5          # bins 0 and 3, 50/50
    got = pick_length_matched(rows, 10, target, random.Random(0))
    assert len(got) == 10
    assert Counter(_bin(len(r["utt"].split())) for r in got) == {0: 5, 3: 5}


def test_frames_unique_and_placeholders_have_pools() -> None:
    for key, frames in FRAMES.items():
        assert len(set(frames)) == len(frames), key
        for f in frames:
            for _, name, _, _ in string.Formatter().parse(f):
                if name:
                    assert name in POOLS[key], (key, f)


def test_kb_pool_questions_are_inlined() -> None:
    for lang in ("en", "ar"):
        assert all("{q" not in f for f in FRAMES[("kb_question", lang)])