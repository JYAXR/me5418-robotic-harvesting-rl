"""Check independent gradient, Adam and counterfactual math on synthetic networks."""

import importlib.util
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from me5418_harvesting.agents.dqn import DQNConfig
from me5418_harvesting.agents.dqn_learning import DQNUpdater, TransitionBatch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gradient_core_test", ROOT / "docs/experiments/018_gradient_diagnostics/gradient_core.py",
)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def state(learner):
    return deepcopy({
        "online": learner.online.state_dict(), "target": learner.target.state_dict(),
        "optimizer": learner.optimizer.state_dict(), "update_count": learner.update_count,
        "modes": [module.training for model in (learner.online, learner.target) for module in model.modules()],
        "gradients": [parameter.grad for parameter in learner.online.parameters()],
    })


def synthetic(*, linear=False, warmup=2, count=2199):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    dtype = torch.float64 if linear else torch.float32
    config = DQNConfig(hidden_sizes=(4, 4), learning_rate=0.01, batch_size=3,
                       target_update_interval=200, gradient_clip_norm=0.05)
    learner = DQNUpdater(3, 3, config)
    if linear:
        learner.online = nn.Linear(3, 3).double()
        with torch.no_grad():
            learner.online.weight.copy_(torch.tensor([[0.7, -0.2, 0.1], [-0.1, 0.5, 0.2], [0.3, -0.4, -0.2]]))
            learner.online.bias.copy_(torch.tensor([0.1, -0.1, 0.2]))
        learner.target = deepcopy(learner.online).requires_grad_(False).eval()
        learner.optimizer = torch.optim.Adam(learner.online.parameters(), lr=config.learning_rate)
    observations = torch.tensor([[1.2, -0.7, 0.3], [-0.4, 0.8, 1.1], [0.6, 0.2, -0.9]], dtype=dtype)
    batch = TransitionBatch(observations, torch.tensor([2, 0, 1]),
                            torch.tensor([1.4, -0.8, 0.2], dtype=dtype), observations.flip(0) * 0.6,
                            torch.tensor([False, True, False]), torch.tensor([False, False, True]))
    returns = torch.tensor([2.1, -1.2, 1.3], dtype=dtype)
    for _ in range(warmup):
        learner.update(batch, demonstration_weight=1.0)
    learner.update_count = count
    probes = [{"name": f"probe_{row}", "observation": observations[row:row + 1].clone(),
               "demonstrated_action": int(batch.actions[row])} for row in range(3)]
    return learner, batch, returns, probes


def independent_losses(learner, batch, returns):
    chosen = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_q = learner.target(batch.next_observations).max(dim=1).values
        target = batch.rewards + learner.config.gamma * torch.where(batch.terminated, 0.0, next_q)
    td = torch.nn.functional.smooth_l1_loss(chosen, target)
    q = learner.online(batch.observations)
    offsets = torch.full_like(q, 0.8)
    offsets[torch.arange(len(q)), batch.actions] = 0
    margin = ((q + offsets).max(dim=1).values - q.gather(1, batch.actions[:, None]).squeeze(1)).mean()
    mc = torch.nn.functional.smooth_l1_loss(chosen, returns.detach())
    return {"td": td, "margin": margin, "mc": mc}, target


def manual_branch(source, batch, returns, name, coefficients):
    learner = deepcopy(source)
    learner.online.train()
    if name == "reset_adam":
        learner.optimizer.state.clear()
    elif name == "zero_first_moment":
        for value in learner.optimizer.state.values():
            value["exp_avg"].zero_()
    learner.optimizer.zero_grad(set_to_none=True)
    before = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    losses, targets = independent_losses(learner, batch, returns)
    terms = [coefficients[key] * losses[key] for key in ("td", "margin", "mc") if coefficients[key]]
    if terms:
        loss = terms[0]
        for term in terms[1:]:
            loss = loss + term
        loss.backward()
    else:
        for parameter in learner.online.parameters():
            parameter.grad = torch.zeros_like(parameter)
    if name != "no_clipping":
        nn.utils.clip_grad_norm_(learner.online.parameters(), learner.config.gradient_clip_norm)
    learner.optimizer.step()
    after = nn.utils.parameters_to_vector(learner.online.parameters()).detach().clone()
    learner.update_count += 1
    if learner.update_count % learner.config.target_update_interval == 0:
        learner.target.load_state_dict(learner.online.state_dict())
    return learner, after.double() - before.double(), targets


@pytest.mark.parametrize("mc_weight", [0, 1])
def test_actual_matches_original_graph_state_clipping_and_absolute_target_clock(mc_weight):
    learner, batch, returns, probes = synthetic()
    expected = deepcopy(learner)
    if mc_weight == 0:
        stats = asdict(expected.update(batch, demonstration_weight=1.0, demonstration_margin=0.8))
    else:
        expected, _, _ = manual_branch(expected, batch, returns, "actual", {"td": 1.0, "margin": 1.0, "mc": 1.0})
    record, actual = core.diagnose_update(learner, batch, returns, probes, mc_weight=mc_weight)
    assert equal(state(actual), state(expected))
    branch = record["counterfactuals"]["actual"]
    assert branch["gradient_norm"] > learner.config.gradient_clip_norm
    assert branch["clip_coefficient"] < 1
    assert branch["target_synchronized"] and actual.update_count == 2200
    assert branch["optimizer_steps_before"]["minimum"] == 2
    assert branch["optimizer_steps_after"]["minimum"] == 3
    if mc_weight == 0:
        for key in ("loss_before", "loss_after", "gradient_norm", "parameter_delta_norm", "target_synchronized"):
            assert branch[key] == stats[key]
        assert branch["losses_before"]["td"] == stats["td_loss_before"]
        assert branch["losses_after_frozen_targets"]["td"] == stats["td_loss_after"]
    assert record["environment_actions"] == record["environment_resets"] == 0
    assert record["independent_optimizer_steps"] == (10 if mc_weight == 0 else 11)
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("mc_weight", [0, 1])
def test_every_counterfactual_starts_from_same_source_and_uses_frozen_targets(mc_weight):
    learner, batch, returns, probes = synthetic()
    record, _ = core.diagnose_update(learner, batch, returns, probes, mc_weight=mc_weight)
    for name, branch in record["counterfactuals"].items():
        if name == "no_mc" and mc_weight == 0:
            assert branch["alias_of"] == "actual" and branch["optimizer_updates"] == 0
            continue
        expected, delta, targets = manual_branch(learner, batch, returns, name, branch["coefficients"])
        assert branch["result_td_update"] == 2200
        assert branch["parameter_delta_norm_float64"] == float(torch.linalg.vector_norm(delta))
        with torch.no_grad():
            chosen = expected.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
            assert branch["losses_after_frozen_targets"]["td"] == float(torch.nn.functional.smooth_l1_loss(chosen, targets))
            for probe, result in zip(probes, branch["probes"], strict=True):
                assert result["q_values"] == expected.online(probe["observation"])[0].tolist()
        steps = branch["optimizer_steps_before"]["minimum"]
        assert steps == (0 if name == "reset_adam" else 2)
        assert branch["optimizer_steps_after"]["minimum"] == steps + 1
    branches = record["counterfactuals"]
    assert branches["zero_current_gradient"]["raw_gradient_norm_float64"] == 0
    assert branches["zero_current_gradient"]["parameter_delta_norm_float64"] > 0
    assert branches["no_clipping"]["clip_coefficient"] == 1
    assert branches["mc_only"]["coefficients"] == {"td": 0, "margin": 0, "mc": 1}


def linear_gradients(learner, batch, returns):
    """Differentiate Huber and max-margin losses directly for a linear Q model."""
    with torch.no_grad():
        q = learner.online(batch.observations)
        chosen = q.gather(1, batch.actions[:, None]).squeeze(1)
        future = learner.target(batch.next_observations).max(dim=1).values
        targets = batch.rewards + learner.config.gamma * torch.where(batch.terminated, 0.0, future)
        vectors = {}
        for name, target in (("td", targets), ("mc", returns)):
            q_derivative = torch.zeros_like(q)
            q_derivative[torch.arange(len(q)), batch.actions] = (chosen - target).clamp(-1, 1) / len(q)
            vectors[name] = torch.cat(((q_derivative.T @ batch.observations).reshape(-1), q_derivative.sum(0)))
        offsets = torch.full_like(q, 0.8)
        offsets[torch.arange(len(q)), batch.actions] = 0
        winners = (q + offsets).argmax(1)
        derivative = torch.zeros_like(q)
        derivative[torch.arange(len(q)), winners] += 1 / len(q)
        derivative[torch.arange(len(q)), batch.actions] -= 1 / len(q)
        vectors["margin"] = torch.cat(((derivative.T @ batch.observations).reshape(-1), derivative.sum(0)))
    return vectors


@pytest.mark.parametrize("mc_weight", [0, 1])
def test_linear_gradients_cosines_probe_derivatives_and_common_denominator_adam_math(mc_weight):
    learner, batch, returns, probes = synthetic(linear=True)
    gradients = linear_gradients(learner, batch, returns)
    record, actual = core.diagnose_update(learner, batch, returns, probes, mc_weight=mc_weight)
    global_stats = record["component_gradients"]["global"]
    for name, gradient in gradients.items():
        assert global_stats["norms"][name] == pytest.approx(float(gradient.norm()), abs=2e-15)
    for pair in global_stats["pairs"]:
        first, second = gradients[pair["first"]], gradients[pair["second"]]
        assert pair["dot"] == pytest.approx(float(first @ second), abs=2e-15)
        assert pair["cosine"] == pytest.approx(float(first @ second / first.norm() / second.norm()), abs=2e-15)
    weights = {"td": 1, "margin": 1, "mc": mc_weight}
    clip = record["counterfactuals"]["actual"]["clip_coefficient"]
    expected_terms = {name: [] for name in ("old_first_moment", "current_td", "current_margin", "current_mc")}
    offset = 0
    for old_parameter, parameter in zip(learner.online.parameters(), actual.online.parameters(), strict=True):
        group = actual.optimizer.param_groups[0]
        beta1, beta2 = group["betas"]
        old_state, new_state = learner.optimizer.state[old_parameter], actual.optimizer.state[parameter]
        step = float(new_state["step"])
        denominator = (new_state["exp_avg_sq"] / (1 - beta2 ** step)).sqrt() + group["eps"]
        scale = -group["lr"] / (1 - beta1 ** step) / denominator.reshape(-1)
        expected_terms["old_first_moment"].append(scale * beta1 * old_state["exp_avg"].reshape(-1))
        for name, gradient in gradients.items():
            expected_terms[f"current_{name}"].append(scale * (1 - beta1) * clip * weights[name]
                                                     * gradient[offset:offset + parameter.numel()])
        offset += parameter.numel()
    expected_terms = {name: torch.cat(values) for name, values in expected_terms.items()}
    decomposition = record["adam_decomposition"]
    for name, vector in expected_terms.items():
        assert decomposition["terms"][name]["norm"] == pytest.approx(float(vector.norm()), abs=3e-15)
    assert decomposition["actual_minus_mathematical_norm"] < 3e-16
    for probe, before, after in zip(probes, record["probes"], record["counterfactuals"]["actual"]["probes"], strict=True):
        q_derivative = torch.zeros(3, dtype=torch.double)
        q_derivative[probe["demonstrated_action"]] = 1
        q_derivative[before["fixed_competitor"]] = -1
        gap_gradient = torch.cat(((q_derivative[:, None] * probe["observation"]).reshape(-1), q_derivative))
        assert before["gap_gradient_norm"] == pytest.approx(float(gap_gradient.norm()), abs=1e-15)
        for name, gradient in gradients.items():
            assert before["negative_component_gradient_projections"][name] == pytest.approx(float(-gap_gradient @ gradient), abs=2e-15)
        for name, displacement in expected_terms.items():
            assert before["adam_term_projections"][name] == pytest.approx(float(gap_gradient @ displacement), abs=3e-15)
        assert sum(before["adam_term_projections"].values()) == pytest.approx(before["adam_mathematical_delta_projection"], abs=1e-15)
        assert abs(after["taylor_residual"]) < 5e-16
        for name in gradients:
            assert sum(layer[name] for layer in before["layer_projections"]) == pytest.approx(before["negative_component_gradient_projections"][name], abs=2e-15)
    assert global_stats["norms"]["mc"] > 0
    if not mc_weight:
        assert decomposition["terms"]["current_mc"]["norm"] == 0


def test_diagnostics_leave_source_gradients_modes_returns_and_rng_unchanged():
    learner, batch, returns, probes = synthetic()
    learner.online.eval()
    learner.online[0].train()
    returns.requires_grad_()
    before = state(learner)
    rng = torch.get_rng_state().clone()
    first, actual = core.diagnose_update(learner, batch, returns, probes, mc_weight=1)
    assert equal(state(learner), before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert returns.grad is None
    second, repeated = core.diagnose_update(learner, batch, returns, iter(probes), mc_weight=1)
    assert first == second
    assert equal(state(actual), state(repeated))
    assert first["source_state_and_rng_unchanged"]


def test_fixed_competitor_taylor_residual_does_not_use_a_switched_competitor():
    learner, _, _, _ = synthetic(linear=True, warmup=0)
    with torch.no_grad():
        learner.online.weight.zero_()
        learner.online.bias.copy_(torch.tensor([0.4, 0.3, 0.2], dtype=torch.double))
    probe = {"name": "switch", "observation": torch.zeros(1, 3, dtype=torch.double), "demonstrated_action": 0}
    before, gradients = core._probe_before(learner, [probe])
    changed = deepcopy(learner)
    initial = nn.utils.parameters_to_vector(changed.online.parameters()).detach().clone()
    with torch.no_grad():
        changed.online.bias.copy_(torch.tensor([0.4, 0.1, 0.35], dtype=torch.double))
    delta = nn.utils.parameters_to_vector(changed.online.parameters()).detach() - initial
    after = core._probe_after(changed, [probe], before, gradients, delta)[0]
    assert after["fixed_competitor"] == 1 and after["best_other_after"] == 2
    assert after["competitor_changed"]
    assert after["fixed_gap"] == pytest.approx(0.3)
    assert after["greedy_gap"] == pytest.approx(0.05)
    assert abs(after["taylor_residual"]) < 1e-15


def test_zero_component_norms_have_null_cosines_and_fresh_adam_is_supported():
    learner, batch, _, probes = synthetic(linear=True, warmup=0)
    with torch.no_grad():
        returns = learner.online(batch.observations).gather(1, batch.actions[:, None]).squeeze(1)
    record, _ = core.diagnose_update(learner, batch, returns, probes, mc_weight=0)
    stats = record["component_gradients"]["global"]
    assert stats["norms"]["mc"] == 0
    assert all(row["cosine"] is None for row in stats["pairs"] if "mc" in (row["first"], row["second"]))
    assert record["adam_decomposition"]["terms"]["old_first_moment"]["norm"] == 0
    assert record["counterfactuals"]["zero_current_gradient"]["parameter_delta_norm_float64"] == 0
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, 0.5, True, "1"])
def test_invalid_coefficients_fail_without_mutation(invalid):
    learner, batch, returns, probes = synthetic()
    before = state(learner)
    with pytest.raises(ValueError, match="weight"):
        core.diagnose_update(learner, batch, returns, probes, mc_weight=invalid)
    assert equal(state(learner), before)


@pytest.mark.parametrize("kind", ["shape", "nan", "dtype", "probe", "duplicate", "weight_decay"])
def test_invalid_targets_probes_or_adam_variant_fail_without_mutation(kind):
    learner, batch, returns, probes = synthetic()
    if kind == "shape":
        returns = returns[:, None]
    elif kind == "nan":
        returns[0] = torch.nan
    elif kind == "dtype":
        returns = returns.double()
    elif kind == "probe":
        probes[0]["observation"] = probes[0]["observation"].repeat(2, 1)
    elif kind == "duplicate":
        probes[0]["name"] = probes[1]["name"]
    elif kind == "weight_decay":
        learner.optimizer.param_groups[0]["weight_decay"] = 0.1
    before = state(learner)
    with pytest.raises(ValueError):
        core.diagnose_update(learner, batch, returns, probes, mc_weight=1)
    assert equal(state(learner), before)


def test_nonfinite_gradient_and_invalid_late_probe_restore_rng_and_source_state():
    learner, batch, returns, probes = synthetic()
    probes[0]["demonstrated_action"] = 30
    before = state(learner)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Probe Q values"):
        core.diagnose_update(learner, batch, returns, probes, mc_weight=1)
    assert equal(state(learner), before) and torch.equal(torch.get_rng_state(), rng)
    probes[0]["demonstrated_action"] = 0
    before = state(learner)
    # Invalid observations exercise the finite-gradient guard after cloning begins.
    batch.observations[0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="gradients"):
        core.diagnose_update(learner, batch, returns, probes, mc_weight=1)
    assert equal(state(learner), before) and torch.equal(torch.get_rng_state(), rng)
