#!/usr/bin/env python
"""Build the routing dataset (synthetic + real MASSIVE out_of_scope).

  python src/route/generate.py [--seed 42] [--out data/route] [--massive-dir data/massive]
                               [--calibration data/kb/calibration_queries.jsonl]
Never reads the MASSIVE test partition.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import string
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from route.frames import FRAMES, POOLS, WRAPPERS  # noqa: E402

LABELS = ("kb_question", "data_lookup", "out_of_scope", "unsafe_request")
LANGS = ("ar", "en")
SPLITS = ("train", "dev", "heldout")
LOCALE = {"ar": "ar-SA", "en": "en-US"}
ROOT = Path(__file__).resolve().parents[2]

_SUPPORT_EN = re.compile(
    r"\b(balance|refund|refunds|card|cards|transfer|account|accounts|ticket|wallet|password|fee|fees|limit|"
    r"bank|payment|pay|money|nile|kyc)\b", re.I)
_SUPPORT_AR = ("رصيد", "استرداد", "بطاق", "تحويل", "حساب", "تذكرة", "تذاكر", "محفظ", "رسوم", "بنك", "دفع",
               "فلوس", "نايل")


@dataclass(frozen=True)
class Caps:
    train: int = 200
    dev: int = 40
    heldout: int = 40


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def frame_id(frame: str) -> str:
    return "f" + hashlib.md5(frame.encode("utf-8")).hexdigest()[:8]


def split_frames(frames: list[str]) -> dict[str, list[str]]:
    if len(frames) < 3:
        raise ValueError("need at least 3 frames to split")
    ordered = sorted(frames, key=lambda f: hashlib.md5(f.encode("utf-8")).hexdigest())
    n_held = max(1, round(len(ordered) * 0.2))
    n_dev = max(1, round(len(ordered) * 0.2))
    return {"heldout": ordered[:n_held], "dev": ordered[n_held:n_held + n_dev], "train": ordered[n_held + n_dev:]}


def expand(template: str, pools: dict[str, list[str]]) -> list[str]:
    names = [f for _, f, _, _ in string.Formatter().parse(template) if f]
    if not names:
        return [template]
    return [template.format(**dict(zip(names, combo))) for combo in itertools.product(*(pools[n] for n in names))]


def wrap(text: str, wrapper: str) -> str:
    if not wrapper:
        return text
    first = text.split()[0]
    if text[:1].isascii() and not first.isupper() and not re.match(r"I\b", text):
        text = text[0].lower() + text[1:]
    return wrapper + text


def candidates(label: str, lang: str, frames: list[str]) -> list[tuple[str, str]]:
    pools, wrappers = POOLS[(label, lang)], WRAPPERS[(label, lang)]
    out, seen = [], set()
    for frame in frames:
        fid = frame_id(frame)
        for filled in expand(frame, pools):
            for w in wrappers:
                text = wrap(filled, w)
                key = norm(text)
                if key not in seen:
                    seen.add(key)
                    out.append((text, fid))
    return out


def synthetic_candidates(label: str, lang: str, split: str) -> list[tuple[str, str]]:
    return candidates(label, lang, split_frames(FRAMES[(label, lang)])[split])


# ----------------------------------------------------------------------------- MASSIVE out_of_scope

def load_massive(path: Path, partition: str) -> list[dict]:
    if partition not in ("train", "dev"):
        raise ValueError("only the train and dev partitions may be read")
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing (run scripts/download_massive.py)")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not {"partition", "utt", "intent"} <= r.keys():
                raise ValueError(f"unexpected MASSIVE row keys: {sorted(r)}")
            if r["partition"] == partition:
                rows.append({"utt": r["utt"], "intent": r["intent"]})
    return rows


def support_like(text: str) -> bool:
    return bool(_SUPPORT_EN.search(text)) or any(s in text for s in _SUPPORT_AR)


def filter_rows(rows: list[dict]) -> tuple[list[dict], int]:
    seen, kept, dropped = set(), [], 0
    for r in rows:
        key = norm(r["utt"])
        if key in seen:
            continue
        seen.add(key)
        if support_like(r["utt"]):
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def pick_stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["intent"], r["utt"])):
        by[r["intent"]].append(r)
    intents = sorted(by)
    for i in intents:
        rng.shuffle(by[i])
    out: list[dict] = []
    while len(out) < n and any(by.values()):
        for i in intents:
            if by[i] and len(out) < n:
                out.append(by[i].pop())
    return out


# ----------------------------------------------------------------------------- build / verify

def build_dataset(massive_dir: Path, seed: int = 42, caps: Caps = Caps()) -> tuple[dict[str, list[dict]], dict]:
    cap_of = {"train": caps.train, "dev": caps.dev, "heldout": caps.heldout}
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    info: dict = {"shortfalls": [], "massive_dropped_support_like": {}, "frames": {}}

    def add(split: str, text: str, label: str, lang: str, source: str, ref: str) -> None:
        out[split].append({"text": text, "label": label, "language": lang, "split": split,
                           "source": source, "ref": ref})

    for lang in LANGS:
        for label in ("kb_question", "data_lookup", "unsafe_request"):
            sp = split_frames(FRAMES[(label, lang)])
            info["frames"][f"{label}/{lang}"] = {s: len(sp[s]) for s in SPLITS}
            for split in SPLITS:
                cands = candidates(label, lang, sp[split])
                random.Random(f"{seed}|{label}|{lang}|{split}").shuffle(cands)
                chosen = cands[: cap_of[split]]
                if len(chosen) < cap_of[split]:
                    info["shortfalls"].append(f"{label}/{lang}/{split}: {len(chosen)} < {cap_of[split]}")
                for text, ref in chosen:
                    add(split, text, label, lang, "synthetic", ref)

        path = massive_dir / f"{LOCALE[lang]}.jsonl"
        train_rows, d1 = filter_rows(load_massive(path, "train"))
        dev_rows, d2 = filter_rows(load_massive(path, "dev"))
        info["massive_dropped_support_like"][lang] = {"train": d1, "dev": d2}
        train_texts = {norm(r["utt"]) for r in train_rows}
        dev_rows = [r for r in dev_rows if norm(r["utt"]) not in train_texts]
        picks = {"train": pick_stratified(train_rows, caps.train, random.Random(f"{seed}|oos|{lang}|train"))}
        picks["heldout"] = pick_stratified(dev_rows, caps.heldout, random.Random(f"{seed}|oos|{lang}|heldout"))
        held = {norm(r["utt"]) for r in picks["heldout"]}
        picks["dev"] = pick_stratified([r for r in dev_rows if norm(r["utt"]) not in held], caps.dev,
                                       random.Random(f"{seed}|oos|{lang}|dev"))
        for split in SPLITS:
            if len(picks[split]) < cap_of[split]:
                info["shortfalls"].append(f"out_of_scope/{lang}/{split}: {len(picks[split])} < {cap_of[split]}")
            for r in picks[split]:
                add(split, r["utt"], "out_of_scope", lang, "massive", r["intent"])

    for split in SPLITS:
        for i, rec in enumerate(out[split]):
            rec["id"] = f"{split}-{i:05d}"
    return out, info


def verify(out: dict[str, list[dict]]) -> None:
    first: dict[str, tuple[str, str]] = {}
    problems = []
    for split in SPLITS:
        for rec in out[split]:
            key = norm(rec["text"])
            if key in first and first[key] != (split, rec["label"]):
                problems.append(f"{rec['text']!r}: {first[key]} vs {(split, rec['label'])}")
            first.setdefault(key, (split, rec["label"]))
        cells = Counter((r["language"], r["label"]) for r in out[split])
        for lang in LANGS:
            for label in LABELS:
                if cells[(lang, label)] == 0:
                    problems.append(f"{split}: no rows for {label}/{lang}")
    if problems:
        raise ValueError("dataset problems:\n" + "\n".join(problems[:20]))


def check_calibration_overlap(out: dict[str, list[dict]], path: Path) -> None:
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "query" not in row:
                raise ValueError(f"{path}: no 'query' key, found {sorted(row)}")
            keys.add(norm(row["query"]))
    hits = [r["text"] for s in SPLITS for r in out[s] if norm(r["text"]) in keys]
    if hits:
        raise ValueError(f"route data overlaps calibration queries: {hits[:5]}")


def write_outputs(out: dict[str, list[dict]], info: dict, out_dir: Path, seed: int, caps: Caps) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        p = out_dir / f"route_{split}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for rec in out[split]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    counts = {s: {lang: dict(Counter(r["label"] for r in out[s] if r["language"] == lang)) for lang in LANGS}
              for s in SPLITS}
    manifest = {"seed": seed, "caps": asdict(caps), "labels": list(LABELS), "counts": counts,
                "frames": info["frames"], "massive_dropped_support_like": info["massive_dropped_support_like"],
                "shortfalls": info["shortfalls"], "sha256": files}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "route")
    ap.add_argument("--massive-dir", type=Path, default=ROOT / "data" / "massive")
    ap.add_argument("--calibration", type=Path, default=ROOT / "data" / "kb" / "calibration_queries.jsonl")
    a = ap.parse_args()

    out, info = build_dataset(a.massive_dir, a.seed)
    verify(out)
    if a.calibration.is_file():
        check_calibration_overlap(out, a.calibration)
    else:
        print(f"WARNING: {a.calibration} not found, overlap check skipped")
    manifest = write_outputs(out, info, a.out, a.seed, Caps())

    for split in SPLITS:
        print(f"{split}: {len(out[split])} rows")
        for lang in LANGS:
            print("   ", lang, manifest["counts"][split][lang])
    print("MASSIVE support-like dropped:", info["massive_dropped_support_like"])
    for s in info["shortfalls"]:
        print("SHORTFALL:", s)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())