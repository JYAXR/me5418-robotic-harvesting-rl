"""Check deterministic interval selection without access to rollout outcomes."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("neighbor023_selection", ROOT / "docs/experiments/023_neighbor_rollouts/run_neighbor_rollouts.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def dense(negative):
    return {"reference_steps": [22], "rows": [[u, None, [negative.get(u, 0.1)], [0]] for u in range(2000, 2013)]}


def test_selects_complete_intervals_and_positive_neighbors():
    result = r.select_windows(dense({2002: -0.1, 2005: -0.1, 2006: -0.2, 2007: -0.1, 2009: -0.8, 2011: -0.1}))
    assert result["anchors"] == [2002, 2006, 2009, 2011]
    assert result["negative_intervals"] == [[2002], [2005, 2006, 2007], [2009], [2011]]
    assert result["window"] == list(range(2001, 2013))
    assert result["selected_negative_count"] == result["all_negative_count"] == 6


def test_longest_interval_tie_selects_earliest():
    result = r.select_windows(dense({2001: -0.1, 2003: -0.1, 2004: -0.1, 2006: -0.1, 2007: -0.1, 2009: -0.8, 2011: -0.1}))
    assert result["anchors"] == [2001, 2003, 2009, 2011]
    assert [2006, 2007] not in result["negative_intervals"]


@pytest.mark.parametrize("case", ["empty_negative", "nonfinite", "missing_update"])
def test_invalid_selection_evidence_is_rejected(case):
    data = dense({2002: -0.1})
    if case == "empty_negative":
        data = dense({})
    elif case == "nonfinite":
        data["rows"][5][2][0] = float("nan")
    else:
        del data["rows"][5]
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        r.select_windows(data)
