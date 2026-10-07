"""Dataset verification tests.

Part 1 tests the verifier itself on tiny synthetic data (no download needed).
Part 2 tests the real MASSIVE files and fails loudly if they are missing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from classifier.dataset import (
    EXPECTED_COUNTS,
    EXPECTED_NUM_INTENTS,
    EXPECTED_NUM_SCENARIOS,
    LOCALES,
    PARTITIONS,
    REQUIRED_FIELDS,
    DatasetError,
    Record,
    by_partition,
    load_locale,
    load_massive,
    verify_dataset,
)

# --------------------------------------------------------------------------- #
# Part 1: verifier logic on synthetic data
# --------------------------------------------------------------------------- #
SMALL_COUNTS = {"train": 4, "dev": 2, "test": 2}
SMALL = dict(
    expected_counts=SMALL_COUNTS, expected_num_intents=4, expected_num_scenarios=2
)


def make_records() -> dict[str, list[Record]]:
    """8 parallel items: 4 intents under 2 scenarios, same ids in both locales."""
    plan = [(p, n) for p, count in SMALL_COUNTS.items() for n in range(count)]
    data: dict[str, list[Record]] = {}
    for locale in LOCALES:
        data[locale] = [
            Record(
                id=str(g),
                locale=locale,
                partition=part,
                scenario=f"s{g % 2}",
                intent=f"s{g % 2}_i{g % 4}",
                utt=f"{locale} text {g}",
            )
            for g, (part, _) in enumerate(plan)
        ]
    return data


def test_synthetic_valid_data_passes() -> None:
    assert verify_dataset(make_records(), **SMALL).passed


def test_wrong_split_count_is_detected() -> None:
    data = make_records()
    data["ar-SA"].remove(next(r for r in data["ar-SA"] if r["partition"] == "train"))
    report = verify_dataset(data, **SMALL)
    assert not report.passed
    assert any("ar-SA/train" in f for f in report.failures)


def test_partition_leakage_is_detected() -> None:
    data = make_records()
    test_row = next(r for r in data["en-US"] if r["partition"] == "test")
    train_id = next(r["id"] for r in data["en-US"] if r["partition"] == "train")
    test_row["id"] = train_id
    report = verify_dataset(data, **SMALL)
    assert report.leakage["en-US"] >= 1
    assert not report.passed


def test_wrong_intent_count_is_detected() -> None:
    report = verify_dataset(make_records(), **{**SMALL, "expected_num_intents": 5})
    assert not report.passed


def test_cross_locale_label_mismatch_is_detected() -> None:
    data = make_records()
    data["en-US"][0]["intent"] = "s0_i2"  # exists in the label set, but differs from ar-SA
    report = verify_dataset(data, **SMALL)
    assert report.label_mismatches and report.label_mismatches > 0
    assert not report.passed


def test_missing_locale_is_detected() -> None:
    data = make_records()
    del data["ar-SA"]
    report = verify_dataset(data, **SMALL)
    assert any("ar-SA" in f for f in report.failures)


def test_loader_rejects_missing_fields(tmp_path: Path) -> None:
    row = {"id": "1", "locale": "en-US", "partition": "train", "intent": "a_b", "utt": "x"}
    (tmp_path / "en-US.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="scenario"):
        load_locale("en-US", tmp_path)


def test_loader_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "en-US.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_locale("en-US", tmp_path)


def test_loader_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="download_massive.py"):
        load_locale("en-US", tmp_path)


# --------------------------------------------------------------------------- #
# Part 2: real MASSIVE data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def massive() -> dict[str, list[Record]]:
    try:
        return load_massive()
    except DatasetError as exc:
        pytest.fail(str(exc), pytrace=False)


@pytest.mark.parametrize("locale", LOCALES)
def test_real_locale_exists(massive, locale: str) -> None:
    assert locale in massive and massive[locale]


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("partition", PARTITIONS)
def test_real_split_counts(massive, locale: str, partition: str) -> None:
    grouped = by_partition(massive[locale])
    assert len(grouped[partition]) == EXPECTED_COUNTS[partition]


@pytest.mark.parametrize("locale", LOCALES)
def test_real_locale_total(massive, locale: str) -> None:
    assert len(massive[locale]) == sum(EXPECTED_COUNTS.values())


@pytest.mark.parametrize("locale", LOCALES)
def test_real_intent_and_scenario_counts(massive, locale: str) -> None:
    assert len({r["intent"] for r in massive[locale]}) == EXPECTED_NUM_INTENTS
    assert len({r["scenario"] for r in massive[locale]}) == EXPECTED_NUM_SCENARIOS


@pytest.mark.parametrize("locale", LOCALES)
def test_real_no_partition_leakage(massive, locale: str) -> None:
    seen: dict[str, str] = {}
    for r in massive[locale]:
        assert seen.setdefault(r["id"], r["partition"]) == r["partition"], r["id"]


@pytest.mark.parametrize("locale", LOCALES)
def test_real_required_fields(massive, locale: str) -> None:
    for r in massive[locale]:
        assert all(isinstance(r[f], str) and r[f] for f in REQUIRED_FIELDS)
        assert r["locale"] == locale


def test_real_labels_consistent_across_locales(massive) -> None:
    ar, en = massive["ar-SA"], massive["en-US"]
    assert {r["intent"] for r in ar} == {r["intent"] for r in en}
    en_label = {r["id"]: r["intent"] for r in en}
    assert all(en_label.get(r["id"]) == r["intent"] for r in ar)


def test_real_full_verification_passes(massive) -> None:
    report = verify_dataset(copy.deepcopy(massive))
    report.raise_if_failed()