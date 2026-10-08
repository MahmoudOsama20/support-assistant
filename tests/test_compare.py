import numpy as np
import pytest

from classifier.compare import (
    align, arabic_buckets, bootstrap_metrics, has_arabic, mcnemar_exact,
    normalize_pred, present_metrics, quirky_qa_share,
)


def test_present_metrics_invalid_and_absent_class_add_no_phantom_class():
    t = np.array([0, 0, 1])
    acc, f1 = present_metrics(t, np.array([0, -1, 1]), 3)
    assert acc == pytest.approx(2 / 3)
    assert f1 == pytest.approx((2 / 3 + 1.0) / 2)
    # predicting a class absent from the truth is wrong but adds no class to the macro
    acc2, f1_2 = present_metrics(t, np.array([0, 2, 1]), 3)
    assert acc2 == pytest.approx(2 / 3)
    assert f1_2 == pytest.approx((2 / 3 + 1.0) / 2)


def test_present_metrics_matches_compute_metrics_present():
    from classifier.metrics import compute_metrics_present
    rng = np.random.default_rng(0)
    t = rng.integers(0, 5, 200)
    p = np.where(rng.random(200) < 0.7, t, rng.integers(-1, 5, 200))
    res = compute_metrics_present(t, p)
    acc = res["accuracy"] if isinstance(res, dict) else res[0]
    f1 = res["macro_f1"] if isinstance(res, dict) else res[1]
    mine = present_metrics(t, p, 5)
    assert mine[0] == pytest.approx(acc)
    assert mine[1] == pytest.approx(f1)


def test_mcnemar_exact():
    t, f = np.ones(10, bool), np.zeros(10, bool)
    assert mcnemar_exact(t, f)[2] == pytest.approx(2 * 0.5**10)
    assert mcnemar_exact(np.array([True, False]), np.array([False, True]))[2] == 1.0
    assert mcnemar_exact(t, t) == (0, 0, 1.0)


def test_normalize_pred():
    assert normalize_pred({"pred": "alarm_set", "method": "exact"}) == "alarm_set"
    assert normalize_pred({"pred": None, "method": "invalid"}) is None
    assert normalize_pred({"pred": "x", "method": "invalid"}) is None
    assert normalize_pred({"pred": ""}) is None


def test_align_rejects_label_disagreement_and_missing_rows():
    k = ("1", "en-US")
    a = {k: {"true": "x", "pred": "x"}}
    with pytest.raises(ValueError, match="disagree"):
        align({"a": a, "b": {k: {"true": "y", "pred": "y"}}}, [k])
    with pytest.raises(ValueError, match="no prediction"):
        align({"a": a, "b": {}}, [k])


def test_bootstrap_brackets_point_estimate_and_paired_clusters():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 4, 100)
    p = np.where(rng.random(100) < 0.8, t, (t + 1) % 4)
    ids = [str(i // 2) for i in range(100)]
    res = bootstrap_metrics(t, {"m": p}, ids, 4, 300, 0)
    acc = present_metrics(t, p, 4)[0]
    lo, hi = np.percentile(res["m"]["acc"], [2.5, 97.5])
    assert lo <= acc <= hi


def test_quirky_qa_share():
    vocab = ["a", "general_quirky", "qa_factoid"]
    t = np.array([0, 1, 2, 0])
    p = np.array([0, 2, 0, 1])   # errors: rows 1,2,3 -> all touch quirky/qa
    r = quirky_qa_share(t, p, vocab)
    assert r["n_errors"] == 3
    assert r["share_of_errors_touching_quirky_or_qa"] == 1.0


def test_arabic_buckets():
    assert has_arabic("وش عنوان email أمي") and not has_arabic("next song")
    keys = [("1", "ar-SA"), ("2", "ar-SA"), ("3", "en-US")]
    utts = {keys[0]: "next song", keys[1]: "قفل الإضاءة", keys[2]: "x"}
    t = np.array([0, 0, 0])
    out = arabic_buckets(keys, utts, t, {"m": np.array([1, 0, 0])})
    assert out["no_arabic"]["n"] == 1 and out["no_arabic"]["accuracy"]["m"] == 0.0
    assert out["has_arabic"]["accuracy"]["m"] == 1.0