"""Verify the five-row supervised-only intervention and exact optimizer updates."""

import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("own004tests", Path(__file__).with_name("run_training.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_only_five_supervised_rows_change():
    q = torch.zeros(1440, 8, requires_grad=True)
    actions = torch.zeros(1440, dtype=torch.long)
    m.core.VERIFIED_WEIGHT = 1.0
    control = m.core.focused_margin_loss(q, actions, 2.0)
    m.core.VERIFIED_WEIGHT = 2.0
    targeted = m.core.focused_margin_loss(q, actions, 2.0)
    torch.testing.assert_close(targeted - control, torch.tensor(5 * 0.8 / 1440))
    (targeted - control).backward()
    assert torch.where(q.grad.abs().sum(1) > 0)[0].tolist() == list(m.core.VERIFIED_ROWS)
    m.core.VERIFIED_WEIGHT = 1.0
    m.prior.core.FOCUS_WEIGHT = 2.0
    assert torch.equal(control, m.prior.core.focused_margin_loss(q, actions, 2.0))


def test_control_and_treatment_full_updates():
    torch.set_num_threads(1)
    source, batch, recovery, _, labels, _ = m.inputs()
    assert [(r["row"], r["action"]) for r in labels] == [(430, 6), (653, 6), (660, 4), (1052, 6), (1057, 2)]
    for weight in (1.0, 2.0):
        m.core.VERIFIED_WEIGHT = weight
        a, ea = m.stages.restore(source)
        b, eb = m.stages.restore(source)
        try:
            actual = m.core.recovery_td_update(a.learner, batch, recovery)
            fn = m.prior.core.recovery_td_update if weight == 1.0 else m.core.independent_update
            expected = fn(b.learner, batch, recovery)
            m.equal(actual, expected, "Exact update metrics")
            m.equal(m.stages.snapshot(a, source["environment_config"]),
                    m.stages.snapshot(b, source["environment_config"]), "Exact updated full state")
            assert a.learner.update_count == 12001
        finally:
            ea.close()
            eb.close()
