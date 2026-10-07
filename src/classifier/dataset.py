"""MASSIVE dataset loading and verification for ar-SA and en-US.

The official partitions (train / dev / test) are used exactly as shipped.
Nothing in this module downloads data; see scripts/download_massive.py.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "massive"

LOCALES: tuple[str, ...] = ("ar-SA", "en-US")
PARTITIONS: tuple[str, ...] = ("train", "dev", "test")
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "locale",
    "partition",
    "scenario",
    "intent",
    "utt",
)

EXPECTED_COUNTS: dict[str, int] = {"train": 11_514, "dev": 2_033, "test": 2_974}
EXPECTED_NUM_INTENTS = 60
EXPECTED_NUM_SCENARIOS = 18

DOWNLOAD_HINT = "Run: python scripts/download_massive.py"


class Record(TypedDict):
    id: str
    locale: str
    partition: str
    scenario: str
    intent: str
    utt: str


class DatasetError(RuntimeError):
    """The dataset files are missing or malformed."""


class DatasetVerificationError(DatasetError):
    """The dataset loaded but failed one or more verification checks."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def locale_path(locale: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / f"{locale}.jsonl"


def load_locale(locale: str, data_dir: Path = DEFAULT_DATA_DIR) -> list[Record]:
    """Load one locale's JSONL file, keeping only the required fields.

    Raises DatasetError with a file/line reference on any structural problem.
    """
    path = locale_path(locale, data_dir)
    if not path.is_file():
        raise DatasetError(f"{path} not found. {DOWNLOAD_HINT}")

    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc})") from exc

            missing = [name for name in REQUIRED_FIELDS if name not in obj]
            if missing:
                raise DatasetError(f"{path}:{lineno}: missing fields {missing}")

            for name in REQUIRED_FIELDS:
                value = str(obj[name]) if name == "id" else obj[name]
                if not isinstance(value, str) or not value.strip():
                    raise DatasetError(
                        f"{path}:{lineno}: field '{name}' must be a non-empty string, "
                        f"got {obj[name]!r}"
                    )

            records.append(
                Record(
                    id=str(obj["id"]),
                    locale=obj["locale"],
                    partition=obj["partition"],
                    scenario=obj["scenario"],
                    intent=obj["intent"],
                    utt=obj["utt"],
                )
            )

    if not records:
        raise DatasetError(f"{path} contains no records")
    return records


def load_massive(
    data_dir: Path = DEFAULT_DATA_DIR,
    locales: Sequence[str] = LOCALES,
) -> dict[str, list[Record]]:
    """Load all requested locales. Returns {locale: [Record, ...]}."""
    return {locale: load_locale(locale, data_dir) for locale in locales}


def by_partition(records: Sequence[Record]) -> dict[str, list[Record]]:
    """Group records by their official partition (train / dev / test)."""
    grouped: dict[str, list[Record]] = {p: [] for p in PARTITIONS}
    for record in records:
        grouped.setdefault(record["partition"], []).append(record)
    return grouped


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass
class VerificationReport:
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    num_intents: dict[str, int] = field(default_factory=dict)
    num_scenarios: dict[str, int] = field(default_factory=dict)
    leakage: dict[str, int] = field(default_factory=dict)
    duplicate_rows: dict[str, int] = field(default_factory=dict)
    union_intents: int = 0
    label_mismatches: int | None = None
    id_set_mismatches: int | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def format(self) -> str:
        lines: list[str] = []
        for locale, parts in self.counts.items():
            lines.append(locale)
            for part, n in parts.items():
                lines.append(f"{part + ':':<7}{n}")
            lines.append(f"{'total:':<7}{sum(parts.values())}")
            lines.append("")
        lines.append(f"Unique intents: {self.union_intents}")
        for locale, n in self.num_intents.items():
            lines.append(f"  {locale}: {n}")
        lines.append(
            "Scenarios: "
            + ", ".join(f"{loc}={n}" for loc, n in self.num_scenarios.items())
        )
        lines.append(f"Partition leakage: {sum(self.leakage.values())}")
        lines.append(
            f"Duplicate (id, partition) rows: {sum(self.duplicate_rows.values())}"
        )
        if self.id_set_mismatches is not None:
            lines.append(f"Cross-locale id mismatches: {self.id_set_mismatches}")
        if self.label_mismatches is not None:
            lines.append(f"Cross-locale label mismatches: {self.label_mismatches}")
        lines.append("")
        lines.append(f"Dataset verification: {'PASSED' if self.passed else 'FAILED'}")
        for failure in self.failures:
            lines.append(f"  - {failure}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if self.failures:
            raise DatasetVerificationError(
                "Dataset verification failed:\n  - " + "\n  - ".join(self.failures)
            )


def verify_dataset(
    records_by_locale: Mapping[str, Sequence[Record]],
    *,
    locales: Sequence[str] = LOCALES,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
    expected_num_intents: int = EXPECTED_NUM_INTENTS,
    expected_num_scenarios: int = EXPECTED_NUM_SCENARIOS,
) -> VerificationReport:
    """Run every check and collect all failures (does not stop at the first)."""
    report = VerificationReport()
    present: list[str] = []

    for locale in locales:
        records = records_by_locale.get(locale)
        if not records:
            report.failures.append(f"{locale}: locale missing or empty")
            continue
        present.append(locale)

        wrong_locale = sum(1 for r in records if r["locale"] != locale)
        if wrong_locale:
            report.failures.append(
                f"{locale}: {wrong_locale} rows have a different 'locale' value"
            )

        unknown = {r["partition"] for r in records} - set(expected_counts)
        if unknown:
            report.failures.append(f"{locale}: unexpected partitions {sorted(unknown)}")

        counter = Counter(r["partition"] for r in records)
        report.counts[locale] = {p: counter.get(p, 0) for p in expected_counts}
        for part, expected in expected_counts.items():
            if counter.get(part, 0) != expected:
                report.failures.append(
                    f"{locale}/{part}: expected {expected} rows, got {counter.get(part, 0)}"
                )

        parts_per_id: dict[str, set[str]] = defaultdict(set)
        for r in records:
            parts_per_id[r["id"]].add(r["partition"])
        leaked = sum(1 for parts in parts_per_id.values() if len(parts) > 1)
        report.leakage[locale] = leaked
        if leaked:
            report.failures.append(
                f"{locale}: {leaked} ids appear in more than one partition"
            )
        report.duplicate_rows[locale] = len(records) - len(
            {(r["id"], r["partition"]) for r in records}
        )

        intents = {r["intent"] for r in records}
        scenarios = {r["scenario"] for r in records}
        report.num_intents[locale] = len(intents)
        report.num_scenarios[locale] = len(scenarios)
        if len(intents) != expected_num_intents:
            report.failures.append(
                f"{locale}: expected {expected_num_intents} intents, got {len(intents)}"
            )
        if len(scenarios) != expected_num_scenarios:
            report.failures.append(
                f"{locale}: expected {expected_num_scenarios} scenarios, got {len(scenarios)}"
            )

    report.union_intents = len(
        {r["intent"] for loc in present for r in records_by_locale[loc]}
    )

    # Cross-locale consistency: MASSIVE is parallel, so ids and labels must agree.
    if len(present) >= 2:
        ref = present[0]
        ref_records = records_by_locale[ref]
        ref_intents = {r["intent"] for r in ref_records}
        ref_label = {r["id"]: r["intent"] for r in ref_records}
        ref_ids = {p: {r["id"] for r in ref_records if r["partition"] == p}
                   for p in expected_counts}
        report.id_set_mismatches = 0
        report.label_mismatches = 0

        for other in present[1:]:
            other_records = records_by_locale[other]
            other_intents = {r["intent"] for r in other_records}
            if other_intents != ref_intents:
                diff = sorted(ref_intents ^ other_intents)
                report.failures.append(
                    f"{ref} vs {other}: intent label sets differ (e.g. {diff[:5]})"
                )
            for part in expected_counts:
                other_ids = {r["id"] for r in other_records if r["partition"] == part}
                diff_n = len(ref_ids[part] ^ other_ids)
                if diff_n:
                    report.id_set_mismatches += diff_n
                    report.failures.append(
                        f"{ref} vs {other}/{part}: {diff_n} ids not shared "
                        f"(partitions are expected to be parallel)"
                    )
            mismatches = sum(
                1 for r in other_records
                if r["id"] in ref_label and ref_label[r["id"]] != r["intent"]
            )
            report.label_mismatches += mismatches
            if mismatches:
                report.failures.append(
                    f"{ref} vs {other}: {mismatches} ids have different intents"
                )

    return report

# --------------------------------------------------------------------------- #
# Labels and splits (Step 2)
# --------------------------------------------------------------------------- #
def build_labels(
    records_by_locale: Mapping[str, Sequence[Record]],
    expected_num: int | None = EXPECTED_NUM_INTENTS,
) -> list[str]:
    """Sorted unique intents from the *train* partitions of all locales.

    Sorted order makes the label ids deterministic. The result is saved with
    the model (labels.json) and is the only label mapping used downstream.
    """
    intents = {
        r["intent"]
        for records in records_by_locale.values()
        for r in records
        if r["partition"] == "train"
    }
    labels = sorted(intents)
    if expected_num is not None and len(labels) != expected_num:
        raise DatasetError(
            f"Expected {expected_num} intents in train, found {len(labels)}"
        )
    return labels


def check_labels_cover(records: Sequence[Record], labels: Sequence[str]) -> None:
    """Fail loudly if any record has an intent that is not in the label map."""
    unseen = sorted({r["intent"] for r in records} - set(labels))
    if unseen:
        raise DatasetError(f"Intents not in label map: {unseen[:10]}")


def select_split(
    records_by_locale: Mapping[str, Sequence[Record]],
    partition: str,
    locales: Sequence[str] = LOCALES,
) -> list[Record]:
    """Concatenate one official partition across locales (deterministic order)."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    selected: list[Record] = []
    for locale in locales:
        selected.extend(r for r in records_by_locale[locale] if r["partition"] == partition)
    return selected


def save_labels(labels: Sequence[str], path: Path) -> None:
    Path(path).write_text(json.dumps(list(labels), ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Label file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))