"""Check exact control equivalence and the three-row supervised-only change."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("focus_test", ROOT / "docs/experiments/034_scene10002_supervision/run_training.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_only_three_margin_rows_change():
    q = torch.zeros(1440, 8, requires_grad=True)
    actions = torch.zeros(1440, dtype=torch.long)
    m.core.FOCUS_WEIGHT = 1.0
    control = m.core.focused_margin_loss(q, actions, 2.0)
    m.core.FOCUS_WEIGHT = 2.0
    targeted = m.core.focused_margin_loss(q, actions, 2.0)
    torch.testing.assert_close(targeted - control, torch.tensor(3 * 0.8 / 1440))
    (targeted - control).backward()
    changed = torch.where(q.grad.abs().sum(1) > 0)[0].tolist()
    assert changed == [98, 102, 103]


@pytest.mark.historical_replay
def test_control_and_independent_updates_preserve_exact_recipe():
    torch.set_num_threads(1)
    source, batch, recovery, _ = m.inputs()
    for weight in (1.0, 2.0):
        m.core.FOCUS_WEIGHT = weight
        a, ea = m.stages.restore(source)
        b, eb = m.stages.restore(source)
        try:
            actual = m.core.recovery_td_update(a.learner, batch, recovery)
            fn = m.stages.core.coverage.recovery_td_update if weight == 1.0 else m.core.independent_update
            expected = fn(b.learner, batch, recovery)
            m.equal(actual, expected, "Exact update metrics")
            m.equal(m.stages.snapshot(a, source["environment_config"]),
                    m.stages.snapshot(b, source["environment_config"]), "Exact updated training state")
            assert a.learner.update_count == 10001
        finally:
            ea.close()
            eb.close()
