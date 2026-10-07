from __future__ import annotations

import pytest

from classifier.dataset import (
    EXPECTED_NUM_INTENTS,
    DatasetError,
    Record,
    build_labels,
    check_labels_cover,
    load_labels,
    load_massive,
    save_labels,
    select_split,
)


def _rec(id_: str, locale: str, partition: str, intent: str) -> Record:
    return Record(id=id_, locale=locale, partition=partition,
                  scenario=intent.split("_")[0], intent=intent, utt="x")


def _data():
    return {
        "ar-SA": [_rec("1", "ar-SA", "train", "b_x"), _rec("2", "ar-SA", "dev", "a_y")],
        "en-US": [_rec("1", "en-US", "train", "a_y"), _rec("2", "en-US", "test", "c_z")],
    }


def test_build_labels_sorted_and_deterministic() -> None:
    data = _data()
    reversed_data = {k: list(reversed(v)) for k, v in data.items()}
    assert build_labels(data, expected_num=2) == ["a_y", "b_x"]
    assert build_labels(reversed_data, expected_num=2) == ["a_y", "b_x"]


def test_build_labels_wrong_count_raises() -> None:
    with pytest.raises(DatasetError, match="Expected 3"):
        build_labels(_data(), expected_num=3)


def test_check_labels_cover_detects_unseen_intent() -> None:
    with pytest.raises(DatasetError, match="c_z"):
        check_labels_cover(select_split(_data(), "test"), ["a_y", "b_x"])


def test_select_split_concatenates_locales_in_order() -> None:
    train = select_split(_data(), "train")
    assert [r["locale"] for r in train] == ["ar-SA", "en-US"]


def test_labels_roundtrip(tmp_path) -> None:
    labels = ["alarm_set", "ترتيب"]
    save_labels(labels, tmp_path / "labels.json")
    assert load_labels(tmp_path / "labels.json") == labels


@pytest.fixture(scope="module")
def real_massive():
    try:
        return load_massive()
    except DatasetError as exc:
        pytest.fail(str(exc), pytrace=False)


@pytest.mark.parametrize("partition,expected", [("train", 23_028), ("dev", 4_066), ("test", 5_948)])
def test_real_combined_split_sizes(real_massive, partition: str, expected: int) -> None:
    assert len(select_split(real_massive, partition)) == expected


def test_real_labels_cover_dev_and_test(real_massive) -> None:
    labels = build_labels(real_massive)
    assert len(labels) == EXPECTED_NUM_INTENTS
    check_labels_cover(select_split(real_massive, "dev") + select_split(real_massive, "test"), labels)