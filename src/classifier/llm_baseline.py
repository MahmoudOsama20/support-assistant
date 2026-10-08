#!/usr/bin/env python
"""Zero-shot LLM intent baseline on MASSIVE via an OpenAI-compatible API (Groq).

Usage:
    export/set GROQ_API_KEY first (never commit it).
    python src/classifier/llm_baseline.py --list-models
    python src/classifier/llm_baseline.py --print-prompt
    python src/classifier/llm_baseline.py --partition dev --n 50      # smoke test on DEV
    python src/classifier/llm_baseline.py                             # final run (TEST, config defaults)

Every response is cached in results/llm_baseline/<model>/<partition>/cache.jsonl, so an
interrupted run resumes by re-running the same command.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier.dataset import LOCALES, PARTITIONS, build_labels, load_massive  # noqa: E402
from classifier.metrics import bootstrap_ci, compute_metrics_present  # noqa: E402

PROMPT_VERSION = "zero_shot_v1"
STRICT_METHODS = ("exact", "normalized")
USER_AGENT = "support-assistant-baseline/0.1"
REQUIRED_KEYS = (
    "base_url", "model", "temperature", "max_tokens", "partition", "n_per_locale",
    "sample_seed", "interval_s", "timeout_s", "max_retries", "output_dir",
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """A non-retryable problem with the LLM API call."""


class RateLimitExhausted(LLMError):
    """The provider asked us to wait too long (probably the daily quota)."""


# --------------------------------------------------------------------------- #
# Prompt, parsing, cache key, sampling (pure functions; unit tested offline)
# --------------------------------------------------------------------------- #
def build_messages(utterance: str, labels: Sequence[str]) -> list[dict[str, str]]:
    system = (
        "You are an intent classifier for a voice assistant. "
        "The user's utterance may be in English or Arabic. "
        "Classify it into exactly ONE of the intents listed below.\n\n"
        "Rules:\n"
        "- Output only the intent label, copied exactly from the list.\n"
        "- Do not translate, explain, or add any other text.\n\n"
        "Intents:\n" + "\n".join(labels)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": utterance}]


def parse_label(raw: str | None, labels: Sequence[str]) -> tuple[str | None, str]:
    """Map a raw model output to one label. Returns (label | None, method).

    method: exact | normalized | contained | fuzzy | invalid.
    """
    if raw is None:
        return None, "invalid"
    text = _THINK_RE.sub("", raw).strip()
    if not text:
        return None, "invalid"

    label_set = set(labels)
    if text in label_set:
        return text, "exact"

    cleaned = text.splitlines()[0].strip().strip("`'\"*.,;: ").lower()
    if cleaned in label_set:
        return cleaned, "normalized"

    lowered = text.lower()
    found = [
        label for label in labels
        if re.search(rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])", lowered)
    ]
    if len(found) == 1:
        return found[0], "contained"
    if len(found) > 1:
        return None, "invalid"  # ambiguous: several labels mentioned

    close = difflib.get_close_matches(cleaned, list(labels), n=1, cutoff=0.8)
    if close:
        return close[0], "fuzzy"
    return None, "invalid"


def cache_key(model: str, messages: Sequence[Mapping[str, str]], temperature: float,
              max_tokens: int) -> str:
    blob = json.dumps(
        {"v": PROMPT_VERSION, "model": model, "messages": list(messages),
         "t": temperature, "m": max_tokens},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sample_ids(records_by_locale: Mapping[str, Sequence[Mapping[str, str]]],
               partition: str, n: int, seed: int) -> list[str]:
    """First n ids of a seeded shuffle of ids present in every locale (nested in n)."""
    id_sets = [
        {r["id"] for r in records_by_locale[loc] if r["partition"] == partition}
        for loc in LOCALES
    ]
    common = sorted(set.intersection(*id_sets), key=lambda s: (len(s), s))
    random.Random(seed).shuffle(common)
    return common[:n]


def latency_stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }


def parse_retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
def _request(url: str, api_key: str, payload: dict | None, timeout: float):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="GET" if payload is None else "POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def list_models(cfg: Mapping[str, Any], api_key: str) -> list[str]:
    url = cfg["base_url"].rstrip("/") + "/models"
    try:
        with _request(url, api_key, None, cfg["timeout_s"]) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"GET /models failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"GET /models failed: {exc}") from exc
    return sorted(m["id"] for m in body.get("data", []))


def chat_once(cfg: Mapping[str, Any], api_key: str,
              messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """One chat completion with retries. Latency covers only the successful attempt."""
    payload = {
        "model": cfg["model"], "messages": list(messages),
        "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"],
    }
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    retries, backoff_total = 0, 0.0

    while True:
        started = time.perf_counter()
        try:
            with _request(url, api_key, payload, cfg["timeout_s"]) as response:
                body = json.loads(response.read().decode("utf-8"))
            latency = time.perf_counter() - started
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            return {
                "raw": (choice.get("message") or {}).get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "latency_s": latency,
                "retries": retries,
                "backoff_s": backoff_total,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            if exc.code in (401, 403):
                raise LLMError(
                    f"HTTP {exc.code}: check GROQ_API_KEY, account access and your network "
                    f"(a blocked request can also return 403). Body: {detail}") from exc
            if exc.code in (400, 404, 422):
                raise LLMError(
                    f"HTTP {exc.code}: bad request or unknown model "
                    f"'{cfg['model']}'. Try --list-models. Body: {detail}") from exc
            if exc.code == 429 or exc.code >= 500:
                wait = parse_retry_after(exc.headers) or min(2 ** retries * 2, 60)
                if exc.code == 429 and wait > 300:
                    raise RateLimitExhausted(
                        f"Rate limit asks for a {wait:.0f}s wait (likely the daily quota). "
                        f"Progress is cached; re-run later to resume.") from exc
            else:
                raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            wait = min(2 ** retries * 2, 60)
            last_error = repr(exc)
        else:  # pragma: no cover
            wait = 2.0

        retries += 1
        if retries > cfg["max_retries"]:
            raise LLMError(f"Giving up after {cfg['max_retries']} retries.")
        time.sleep(wait)
        backoff_total += wait


# --------------------------------------------------------------------------- #
# Cache and run loop
# --------------------------------------------------------------------------- #
def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    cache[row["key"]] = row
    return cache


def run_calls(cfg: Mapping[str, Any], api_key: str, labels: Sequence[str],
              index: Mapping[tuple[str, str], Mapping[str, str]], ids: Sequence[str],
              cache: dict[str, dict[str, Any]], cache_path: Path) -> int:
    """Call the API for every uncached (id, locale). Returns the number of new calls."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_start, new_calls = 0.0, 0
    for id_ in tqdm(ids, desc="utterance ids"):
        for locale in LOCALES:
            record = index[(locale, id_)]
            messages = build_messages(record["utt"], labels)
            key = cache_key(cfg["model"], messages, cfg["temperature"], cfg["max_tokens"])
            if key in cache:
                continue
            pause = cfg["interval_s"] - (time.monotonic() - last_start)
            if pause > 0:
                time.sleep(pause)
            last_start = time.monotonic()

            result = chat_once(cfg, api_key, messages)
            row = {
                "key": key, "id": id_, "locale": locale, "utt": record["utt"],
                "model": cfg["model"], "prompt_version": PROMPT_VERSION,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **result,
            }
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache[key] = row
            new_calls += 1
    return new_calls


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def scope_metrics(rows: Sequence[Mapping[str, Any]], labels: Sequence[str],
                  n_boot: int) -> dict[str, Any]:
    label2id = {label: i for i, label in enumerate(labels)}
    y_true = [label2id[r["true"]] for r in rows]
    y_pred = [label2id[r["pred"]] if r["pred"] is not None else -1 for r in rows]
    out: dict[str, Any] = {"n": len(rows), **compute_metrics_present(y_true, y_pred)}
    out["strict_accuracy"] = float(np.mean([
        r["pred"] == r["true"] and r["method"] in STRICT_METHODS for r in rows]))
    methods = Counter(r["method"] for r in rows)
    out["invalid_rate"] = methods["invalid"] / len(rows)
    out["fuzzy_rate"] = methods["fuzzy"] / len(rows)
    out["contained_rate"] = methods["contained"] / len(rows)
    if n_boot:
        out["ci95"] = bootstrap_ci(y_true, y_pred, n_boot=n_boot, present_only=True)
    return out


def evaluate_sample(cfg: Mapping[str, Any], labels: Sequence[str],
                    index: Mapping[tuple[str, str], Mapping[str, str]], ids: Sequence[str],
                    cache: Mapping[str, Mapping[str, Any]], n_boot: int
                    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing = 0
    for id_ in ids:
        for locale in LOCALES:
            record = index[(locale, id_)]
            messages = build_messages(record["utt"], labels)
            cached = cache.get(cache_key(cfg["model"], messages, cfg["temperature"],
                                         cfg["max_tokens"]))
            if cached is None:
                missing += 1
                continue
            pred, method = parse_label(cached["raw"], labels)
            rows.append({
                "id": id_, "locale": locale, "utt": record["utt"], "true": record["intent"],
                "raw": cached["raw"], "pred": pred, "method": method,
                "finish_reason": cached.get("finish_reason"),
                "latency_s": cached["latency_s"],
                "prompt_tokens": cached["prompt_tokens"],
                "completion_tokens": cached["completion_tokens"],
                "retries": cached.get("retries", 0), "backoff_s": cached.get("backoff_s", 0.0),
            })

    summary: dict[str, Any] = {
        "model": cfg["model"], "prompt_version": PROMPT_VERSION,
        "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"],
        "sampling": {"partition": cfg["partition"], "seed": cfg["sample_seed"],
                     "n_ids": len(ids)},
        "n_scored": len(rows), "n_missing": missing,
    }
    if not rows:
        return summary, rows

    scopes = {"overall": list(rows)}
    for locale in LOCALES:
        scopes[locale] = [r for r in rows if r["locale"] == locale]
    summary["metrics"] = {k: scope_metrics(v, labels, n_boot) for k, v in scopes.items() if v}
    summary["parse_methods"] = {k: dict(Counter(r["method"] for r in v))
                                for k, v in scopes.items() if v}
    summary["latency_s"] = {k: latency_stats([r["latency_s"] for r in v])
                            for k, v in scopes.items() if v}

    mean_in = float(np.mean([r["prompt_tokens"] for r in rows]))
    mean_out = float(np.mean([r["completion_tokens"] for r in rows]))
    cost = None
    price_in, price_out = cfg.get("price_per_1m_input_usd"), cfg.get("price_per_1m_output_usd")
    if price_in is not None and price_out is not None:
        cost = (mean_in * price_in + mean_out * price_out) / 1e6
    summary["tokens"] = {
        "mean_prompt": mean_in, "mean_completion": mean_out,
        "total_prompt": int(sum(r["prompt_tokens"] for r in rows)),
        "total_completion": int(sum(r["completion_tokens"] for r in rows)),
        "truncated_outputs": sum(r["finish_reason"] == "length" for r in rows),
    }
    summary["cost"] = {
        "list_price_usd_per_request": cost,
        "price_per_1m_input_usd": price_in, "price_per_1m_output_usd": price_out,
        "price_date": cfg.get("price_date"),
        "note": "list-price equivalent; actual cost on the free tier is $0",
    }
    summary["retries"] = {
        "calls_with_retries": sum(r["retries"] > 0 for r in rows),
        "total_retries": int(sum(r["retries"] for r in rows)),
        "total_backoff_s": float(sum(r["backoff_s"] for r in rows)),
    }
    errors = Counter((r["true"], r["pred"]) for r in rows
                     if r["pred"] is not None and r["pred"] != r["true"])
    summary["top_confused_pairs"] = [
        {"true": t, "pred": p, "count": c} for (t, p), c in errors.most_common(10)]
    summary["environment"] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return summary, rows


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"\nmodel={summary['model']} prompt={summary['prompt_version']} "
          f"partition={summary['sampling']['partition']} ids={summary['sampling']['n_ids']} "
          f"scored={summary['n_scored']} missing={summary['n_missing']}")
    if "metrics" not in summary:
        print("No calls scored yet.")
        return
    print(f"{'scope':<9}{'n':>6}{'acc':>8}{'macro-F1':>10}{'strict':>8}{'invalid':>9}{'fuzzy':>7}"
          f"   95% CI acc / macro-F1")
    for scope, m in summary["metrics"].items():
        ci = m.get("ci95")
        ci_s = (f"[{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}] / "
                f"[{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]") if ci else "-"
        print(f"{scope:<9}{m['n']:>6}{m['accuracy']:>8.4f}{m['macro_f1']:>10.4f}"
              f"{m['strict_accuracy']:>8.4f}{m['invalid_rate']:>9.2%}{m['fuzzy_rate']:>7.1%}   {ci_s}")
    lat = summary["latency_s"]["overall"]
    tok = summary["tokens"]
    print(f"latency (successful attempt): p50={lat['p50']:.3f}s p95={lat['p95']:.3f}s "
          f"mean={lat['mean']:.3f}s")
    print(f"tokens/request: prompt={tok['mean_prompt']:.0f} completion={tok['mean_completion']:.1f} "
          f"truncated={tok['truncated_outputs']}")
    r = summary["retries"]
    print(f"retries: {r['calls_with_retries']} calls, {r['total_retries']} retries, "
          f"{r['total_backoff_s']:.0f}s backoff (excluded from latency)")
    print(f"parse methods (overall): {summary['parse_methods']['overall']}")
    c = summary["cost"]["list_price_usd_per_request"]
    print("list-price cost/request: " + (f"${c:.6f}" if c is not None else
          "not computed (set prices in configs/llm_baseline.yaml)"))
    print("top confusions: " + "; ".join(
        f"{p['true']}->{p['pred']} ({p['count']})" for p in summary["top_confused_pairs"][:5]))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    overrides = {"model": args.model, "n_per_locale": args.n,
                 "partition": args.partition, "interval_s": args.interval}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"Config {path} is missing keys: {missing}")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "llm_baseline.yaml")
    parser.add_argument("--model")
    parser.add_argument("--n", type=int, help="ids per locale (2 calls per id)")
    parser.add_argument("--partition", choices=PARTITIONS)
    parser.add_argument("--interval", type=float, help="minimum seconds between calls")
    parser.add_argument("--bootstrap", type=int, default=1000, help="0 disables CIs")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true", help="score the cache, no API calls")
    args = parser.parse_args()
    cfg = load_config(args.config, args)

    api_key = os.environ.get("GROQ_API_KEY", "")
    try:
        if args.list_models:
            if not api_key:
                raise SystemExit("Set GROQ_API_KEY first.")
            print("\n".join(list_models(cfg, api_key)))
            return 0

        records = load_massive()
        labels = build_labels(records)

        if args.print_prompt:
            system = build_messages("<utterance>", labels)[0]["content"]
            print(system)
            print(f"\n[{len(labels)} labels, {len(system)} characters, "
                  f"roughly {len(system) // 4} tokens (rough estimate; real counts come from the API)]")
            return 0

        partition = cfg["partition"]
        index = {(loc, r["id"]): r for loc in LOCALES for r in records[loc]
                 if r["partition"] == partition}
        ids = sample_ids(records, partition, cfg["n_per_locale"], cfg["sample_seed"])
        slug = cfg["model"].replace("/", "_")
        out_dir = PROJECT_ROOT / cfg["output_dir"] / slug / partition
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / "cache.jsonl"
        cache = load_cache(cache_path)
        (out_dir / "sample_ids.json").write_text(json.dumps(ids), encoding="utf-8")

        if partition == "test" and not args.evaluate_only:
            print(f"NOTE: running on TEST with frozen prompt {PROMPT_VERSION}. "
                  f"Do not edit the prompt based on these results; iterate on --partition dev.")

        if not args.evaluate_only:
            if not api_key:
                raise SystemExit("Set GROQ_API_KEY first (see the run instructions).")
            total = len(ids) * len(LOCALES)
            todo = sum(
                cache_key(cfg["model"], build_messages(index[(loc, i)]["utt"], labels),
                          cfg["temperature"], cfg["max_tokens"]) not in cache
                for i in ids for loc in LOCALES)
            print(f"{total} calls planned, {total - todo} cached, {todo} to run "
                  f"(~{todo * cfg['interval_s'] / 60:.0f} min at {cfg['interval_s']}s/call, "
                  f"longer if rate-limited).")
            try:
                run_calls(cfg, api_key, labels, index, ids, cache, cache_path)
            except RateLimitExhausted as exc:
                print(f"\nSTOPPED: {exc}")
    except LLMError as exc:
        print(f"\nERROR: {exc}\nProgress so far is cached; fix the problem and re-run to resume.",
              file=sys.stderr)
        return 1

    summary, rows = evaluate_sample(cfg, labels, index, ids, cache, args.bootstrap)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print_summary(summary)
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())