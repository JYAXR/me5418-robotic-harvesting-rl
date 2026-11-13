"""Independently verify retrospective objective, gradient and Adam diagnostics."""

import argparse
import hashlib
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from me5418_harvesting.agents.dqn import DQNConfig, build_q_network

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anchor = load_module("verify018_anchor", ROOT / "docs/experiments/017_mc_anchor/verify_mc_anchor.py")
require, require_equal = anchor.require, anchor.require_equal
CASES = (2001, 2008, 2009, 2010, 2018, 2152, 2158, 2161, 2200, 2201, 3000, 4500, 8500, 10000)
INDEPENDENT_CASES = (2008, 2201, 8500)
REFERENCE_PROBES = ((10001, 6, 40), (10001, 18, 52), (10001, 20, 54), (10001, 21, 55),
                    (10001, 22, 56), (10001, 23, 57), (10001, 24, 58), (10002, 37, 112))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label, *, atol=1e-10, rtol=1e-8):
    """Compare independently derived scalars while rejecting nonfinite values."""
    require(math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), label)


def finite_tree(value, path="record"):
    """Reject silent NaN and infinity anywhere in persisted diagnostic evidence."""
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"finite numeric value: {path}")


def vector(tensors):
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


def norm(value):
    return float(torch.linalg.vector_norm(value.double()))


def cosine(left, right):
    denominator = norm(left) * norm(right)
    return None if denominator == 0 else float(torch.dot(left.double(), right.double())) / denominator


def independent_components(learner, batch, returns, weight):
    """Derive all objectives and gradients without the new diagnostic implementation."""
    fixed = anchor.validate_anchor(returns, batch, weight)
    model = deepcopy(learner)
    model.online.train()
    model.optimizer.zero_grad(set_to_none=True)
    parameters = tuple(model.online.parameters())
    first_values = model.online(batch.observations)
    chosen = first_values.gather(1, batch.actions[:, None]).squeeze(1)
    with torch.no_grad():
        future = model.target(batch.next_observations).max(dim=1).values
        future = torch.where(batch.terminated, torch.zeros_like(future), future)
        targets = batch.rewards + model.config.gamma * future
    td = F.smooth_l1_loss(chosen, targets)
    margin_values = model.online(batch.observations)
    offsets = torch.full_like(margin_values, 0.8)
    offsets.scatter_(1, batch.actions[:, None], 0.0)
    margin = ((margin_values + offsets).max(dim=1).values
              - margin_values.gather(1, batch.actions[:, None]).squeeze(1)).mean()
    mc = F.smooth_l1_loss(chosen, fixed)
    losses = {"td": td, "margin": margin, "mc": mc}
    gradients = {name: torch.autograd.grad(loss, parameters, retain_graph=True)
                 for name, loss in losses.items()}
    total = td + 1.0 * margin
    if weight:
        total = total + weight * mc
    total_gradients = torch.autograd.grad(total, parameters)
    # Match the original per-parameter norm reduction, rather than flattening first.
    raw_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(gradient, 2) for gradient in total_gradients]), 2)
    coefficient = torch.clamp(model.config.gradient_clip_norm / (raw_norm + 1e-6), max=1.0)
    return {
        "losses": {**{name: float(value.detach()) for name, value in losses.items()}, "total": float(total.detach())},
        "gradients": {name: vector(values) for name, values in gradients.items()},
        "gradient_tensors": gradients, "total_gradient_tensors": total_gradients,
        "total_gradient": vector(total_gradients), "gradient_norm": float(raw_norm),
        "clip_coefficient": float(coefficient), "clip_tensor": coefficient,
        "targets": targets.detach(), "returns": fixed,
    }


def independent_branch(learner, batch, returns, weight, branch="actual"):
    """Execute an isolated branch using the already independent attempt-017 updater."""
    if branch not in ("actual", "reset_adam", "zero_first_moment"):
        raise ValueError("Only the three declared independent branches are supported.")
    model = deepcopy(learner)
    if branch == "reset_adam":
        model.optimizer.state.clear()
    elif branch == "zero_first_moment":
        for state in model.optimizer.state.values():
            state["exp_avg"].zero_()
    stats = anchor.independent_update(model, batch, returns, weight)
    return model, stats


def independent_adam_parts(before, after, components, weight):
    """Separate inherited momentum and present objectives with the actual new denominator."""
    require(len(before.optimizer.param_groups) == len(after.optimizer.param_groups) == 1,
            "one historical Adam parameter group")
    group = before.optimizer.param_groups[0]
    require(group["weight_decay"] == 0 and not group["amsgrad"] and not group["maximize"],
            "ordinary historical Adam without decay, AMSGrad or maximization")
    beta1, beta2 = group["betas"]
    parts = {name: [] for name in ("old_first_moment", "current_td", "current_margin", "current_mc", "rounding_remainder")}
    details = []
    parameter_before = tuple(before.online.parameters())
    parameter_after = tuple(after.online.parameters())
    for index, (old_parameter, new_parameter) in enumerate(zip(parameter_before, parameter_after, strict=True)):
        old_state = before.optimizer.state[old_parameter]
        new_state = after.optimizer.state[new_parameter]
        step = float(new_state["step"])
        require(step == float(old_state["step"]) + 1, "one preserved Adam bias-correction step")
        denominator = new_state["exp_avg_sq"].double().sqrt() / math.sqrt(1 - beta2 ** step) + group["eps"]
        scale = -group["lr"] / (1 - beta1 ** step) / denominator
        moments = {"old_first_moment": beta1 * old_state["exp_avg"].double()}
        for name in ("td", "margin", "mc"):
            coefficient = weight if name == "mc" else 1.0
            raw = components["gradient_tensors"][name][index].double()
            moments[f"current_{name}"] = ((1 - beta1) * components["clip_coefficient"] * coefficient * raw)
        moments["rounding_remainder"] = new_state["exp_avg"].double() - sum(moments.values())
        for name, moment in moments.items():
            parts[name].append(scale * moment)
        details.append({"step": step, "beta1": beta1, "beta2": beta2, "learning_rate": group["lr"],
                        "epsilon": group["eps"], "denominator_min": float(denominator.min()),
                        "denominator_max": float(denominator.max())})
    result = {name: vector(values) for name, values in parts.items()}
    delta = vector(parameter_after).double() - vector(parameter_before).double()
    predicted = sum(result.values())
    result.update(actual_delta=delta, reconstructed_delta=predicted, numerical_remainder=delta - predicted)
    result["parameter_details"] = details
    return result


def independent_probe(learner, probe):
    """Differentiate the batch-one reference-action gap with its competitor held fixed."""
    model = deepcopy(learner.online)
    model.eval()
    values = model(probe["observation"])[0]
    action = probe["demonstrated_action"]
    competitors = values.detach().clone()
    competitors[action] = -torch.inf
    competitor = int(competitors.argmax())
    gradients = torch.autograd.grad(values[action] - values[competitor], tuple(model.parameters()))
    return {"q_values": values.detach().tolist(), "greedy_action": int(values.argmax()),
            "reference_action": action, "competitor": competitor,
            "gap": float(values[action].detach()) - float(values[competitor].detach()),
            "gradient": vector(gradients)}


def independent_after_probe(learner, probe, competitor):
    with torch.no_grad():
        values = learner.online(probe["observation"])[0]
    action = probe["demonstrated_action"]
    other = values.clone()
    other[action] = -torch.inf
    return {"q_values": values.tolist(), "greedy_action": int(values.argmax()),
            "competitor": int(other.argmax()),
            "fixed_gap": float(values[action]) - float(values[competitor]),
            "max_other_gap": float(values[action]) - float(other.max())}


def verify_rounding_envelope(after_parameters, mathematical_delta, residual, absolute_term_sum):
    """Bound stored-weight rounding at each coordinate's own float32 scale.

    Half the larger adjacent spacing bounds the final round-to-nearest store.
    The ordinary Adam path then gets gamma(8) for its square root, scalar casts,
    denominator arithmetic and update division/multiplication. The measured
    post-update moments are the inputs, so moment accumulation is not counted
    again. A float64 allowance covers reconstructing and adding the five terms.
    No tolerance is estimated from the observed experiment residuals.
    """
    parameters = vector(tuple(after_parameters))
    require(parameters.dtype == torch.float32 and parameters.shape == mathematical_delta.shape == residual.shape
            == absolute_term_sum.shape and bool(torch.isfinite(parameters).all()),
            "finite float32 parameters and matching Adam residual vectors")
    upward = torch.nextafter(parameters, torch.full_like(parameters, torch.inf)) - parameters
    downward = parameters - torch.nextafter(parameters, torch.full_like(parameters, -torch.inf))
    half_spacing = 0.5 * torch.maximum(upward.abs(), downward.abs()).double()
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    gamma_eight = 8 * unit_roundoff / (1 - 8 * unit_roundoff)
    smallest = float(torch.nextafter(torch.tensor(0.0), torch.tensor(torch.inf)))
    arithmetic = gamma_eight * mathematical_delta.abs()
    reconstruction = 32 * torch.finfo(torch.float64).eps * absolute_term_sum
    bound = half_spacing + arithmetic + reconstruction + 8 * smallest
    require(bool(torch.isfinite(bound).all()) and bool((residual.abs() <= bound).all()),
            "Adam residual fits scale-aware float32 storage and arithmetic envelope")
    return {"passed": True, "parameter_dtype": "float32", "parameter_elements": parameters.numel(),
            "unit_roundoff": unit_roundoff, "adam_arithmetic_gamma8": gamma_eight,
            "storage_half_spacing_max": float(half_spacing.max()),
            "storage_half_spacing_norm": norm(half_spacing),
            "coordinate_envelope_max": float(bound.max()), "coordinate_envelope_norm": norm(bound),
            "residual_max": float(residual.abs().max()), "residual_norm": norm(residual),
            "maximum_coordinate_fraction_of_envelope": float((residual.abs() / bound).max()),
            "semantics": "Coordinate-wise half-ULP storage bound plus gamma(8) Adam arithmetic and float64 reconstruction allowances; no fitted absolute residual threshold."}


def verify_alignment(record, vectors, label):
    require(set(record["norms"]) == set(vectors), f"{label} component identities")
    for name, value in vectors.items():
        close(record["norms"][name], norm(value), f"{label} {name} norm")
    expected_pairs = [("td", "margin"), ("td", "mc"), ("margin", "mc")]
    require([(row["first"], row["second"]) for row in record["pairs"]] == expected_pairs,
            f"{label} complete component pairs")
    for row in record["pairs"]:
        left, right = vectors[row["first"]], vectors[row["second"]]
        close(row["dot"], float(left.double() @ right.double()), f"{label} gradient dot product")
        expected = cosine(left, right)
        if expected is None:
            require(row["cosine"] is None, f"{label} undefined zero-norm cosine")
        else:
            close(row["cosine"], expected, f"{label} cosine")


def parameter_slices(learner):
    offset, parameters, layers = 0, [], {}
    for name, parameter in learner.online.named_parameters():
        section = slice(offset, offset + parameter.numel())
        parameters.append((name, list(parameter.shape), section))
        layers.setdefault(name.rsplit(".", 1)[0], []).append(section)
        offset += parameter.numel()
    return parameters, layers


def selected_vector(value, sections):
    return torch.cat([value[section] for section in sections])


def verify_numeric_case(record, learner, batch, returns, probes, numerical_audit=None):
    """Recalculate selected-case mathematics and three branches independently."""
    weight = record["mc_weight"]
    components = independent_components(learner, batch, returns, weight)
    gradients = {name: value.double() for name, value in components["gradients"].items()}
    actual_gradient = components["total_gradient"].double()
    weighted = gradients["td"] + gradients["margin"] + weight * gradients["mc"]
    require(record["losses_before"] == {name: components["losses"][name] for name in gradients},
            "independently reconstructed component losses")
    statistics = record["component_gradients"]
    verify_alignment(statistics["global"], gradients, "global")
    close(statistics["actual_gradient_norm"], norm(actual_gradient), "actual gradient norm")
    close(statistics["weighted_component_sum_norm"], norm(weighted), "weighted component gradient norm")
    close(statistics["actual_minus_component_sum_norm"], norm(actual_gradient - weighted), "gradient accumulation residual")
    parameters, layers = parameter_slices(learner)
    require(record["parameter_count"] == len(parameters)
            and record["parameter_elements"] == actual_gradient.numel(), "complete parameter layout")
    require(len(statistics["parameters"]) == len(parameters) and len(statistics["layers"]) == len(layers),
            "complete parameter and layer gradient diagnostics")
    for row, (name, shape, section) in zip(statistics["parameters"], parameters, strict=True):
        require(row["name"] == name and row["shape"] == shape, "gradient parameter identity")
        verify_alignment(row, {key: value[section] for key, value in gradients.items()}, name)
    for row, (name, sections) in zip(statistics["layers"], layers.items(), strict=True):
        require(row["name"] == name, "gradient layer identity")
        verify_alignment(row, {key: selected_vector(value, sections) for key, value in gradients.items()}, name)
    models, stats = {}, {}
    for branch in ("actual", "reset_adam", "zero_first_moment"):
        models[branch], stats[branch] = independent_branch(learner, batch, returns, weight, branch)
    parts = independent_adam_parts(learner, models["actual"], components, weight)
    decomposition = record["adam_decomposition"]
    terms = ("old_first_moment", "current_td", "current_margin", "current_mc", "rounding_remainder")
    rounding_check = verify_rounding_envelope(
        models["actual"].online.parameters(), parts["reconstructed_delta"], parts["numerical_remainder"],
        sum(parts[name].abs() for name in terms))
    if numerical_audit is not None:
        numerical_audit["adam_rounding_envelope"] = rounding_check
    require(set(decomposition["terms"]) == set(terms), "all five additive Adam moment terms")
    close(decomposition["common_clip_coefficient"], components["clip_coefficient"], "shared Adam clipping coefficient")
    clipped = vector([gradient * components["clip_tensor"]
                      for gradient in components["total_gradient_tensors"]]).double()
    for name in terms:
        close(decomposition["terms"][name]["norm"], norm(parts[name]), f"Adam {name} norm")
    close(decomposition["gradient_sum_residual_norm"], norm(actual_gradient - weighted), "Adam gradient sum residual")
    close(decomposition["clip_rounding_residual_norm"],
          norm(clipped - components["clip_coefficient"] * actual_gradient), "clipping arithmetic residual")
    close(decomposition["mathematical_delta_norm"], norm(parts["reconstructed_delta"]), "mathematical Adam displacement")
    close(decomposition["actual_delta_norm"], norm(parts["actual_delta"]), "actual Adam displacement")
    close(decomposition["actual_minus_mathematical_norm"], norm(parts["numerical_remainder"]), "parameter arithmetic residual")
    close(decomposition["actual_minus_mathematical_max_abs"], float(parts["numerical_remainder"].abs().max()),
          "maximum parameter arithmetic residual")
    require(len(decomposition["parameter_details"]) == len(parameters), "every Adam parameter denominator")
    for row, expected, (name, _, _) in zip(decomposition["parameter_details"], parts["parameter_details"], parameters, strict=True):
        require(row["name"] == name, "Adam parameter identity")
        for key, value in expected.items():
            close(row[key], value, f"Adam {name} {key}")
    before_parameters = vector(tuple(learner.online.parameters())).double()
    for branch, model in models.items():
        evidence, expected = record["counterfactuals"][branch], stats[branch]
        for key in ("loss_before", "loss_after", "gradient_norm", "parameter_delta_norm", "target_synchronized"):
            require(evidence[key] == expected[key], f"independent {branch} exact update diagnostic: {key}")
        require(evidence["losses_before"] == {"td": expected["td_loss_before"],
                                              "margin": expected["demonstration_loss_before"],
                                              "mc": expected["return_loss_before"]}, "branch pre-loss components")
        require(evidence["losses_after_frozen_targets"] == {"td": expected["td_loss_after"],
                                                            "margin": expected["demonstration_loss_after"],
                                                            "mc": expected["return_loss_after"]}, "branch frozen-target post-loss components")
        delta = vector(tuple(model.online.parameters())).double() - before_parameters
        close(evidence["parameter_delta_norm_float64"], norm(delta), f"{branch} displacement norm")
    for probe, evidence in zip(probes, record["probes"], strict=True):
        expected = independent_probe(learner, probe)
        derivative = expected["gradient"].double()
        require(evidence["name"] == probe["name"] and evidence["q_values"] == expected["q_values"]
                and evidence["fixed_competitor"] == expected["competitor"]
                and evidence["greedy_action"] == expected["greedy_action"]
                and evidence["fixed_gap"] == expected["gap"], "independent batch-one initial Q values and competitor")
        close(evidence["gap_gradient_norm"], norm(derivative), "independent probe Jacobian norm")
        for name, gradient in gradients.items():
            projection = -float(derivative @ gradient)
            close(evidence["negative_component_gradient_projections"][name], projection, "negative component projection")
            close(evidence["weighted_negative_component_gradient_projections"][name],
                  (weight if name == "mc" else 1) * projection, "weighted negative component projection")
        close(evidence["negative_actual_gradient_projection"], -float(derivative @ actual_gradient), "negative actual gradient projection")
        close(evidence["negative_clipped_gradient_projection"], -float(derivative @ clipped), "negative clipped gradient projection")
        all_gradients = {**gradients, "actual": actual_gradient}
        require(len(evidence["parameter_projections"]) == len(parameters)
                and len(evidence["layer_projections"]) == len(layers), "complete localized gap projections")
        for row, (name, _, section) in zip(evidence["parameter_projections"], parameters, strict=True):
            require(row["name"] == name, "probe parameter identity")
            close(row["gap_gradient_norm"], norm(derivative[section]), "parameter gap norm")
            for key, gradient in all_gradients.items():
                close(row[key], -float(derivative[section] @ gradient[section]), "parameter gap projection")
        for row, (name, sections) in zip(evidence["layer_projections"], layers.items(), strict=True):
            require(row["name"] == name, "probe layer identity")
            local_derivative = selected_vector(derivative, sections)
            close(row["gap_gradient_norm"], norm(local_derivative), "layer gap norm")
            for key, gradient in all_gradients.items():
                close(row[key], -float(local_derivative @ selected_vector(gradient, sections)), "layer gap projection")
        for name in terms:
            close(evidence["adam_term_projections"][name], float(derivative @ parts[name]), "Adam term projection")
        close(evidence["adam_mathematical_delta_projection"], float(derivative @ parts["reconstructed_delta"]), "Adam mathematical projection")
        close(evidence["actual_delta_projection"], float(derivative @ parts["actual_delta"]), "actual displacement projection")
        for branch, model in models.items():
            after = next(row for row in record["counterfactuals"][branch]["probes"] if row["name"] == probe["name"])
            expected_after = independent_after_probe(model, probe, expected["competitor"])
            require(after["q_values"] == expected_after["q_values"]
                    and after["greedy_action"] == expected_after["greedy_action"]
                    and after["best_other_after"] == expected_after["competitor"]
                    and after["fixed_gap"] == expected_after["fixed_gap"]
                    and after["greedy_gap"] == expected_after["max_other_gap"], "independent exact counterfactual probe outputs")
            delta = vector(tuple(model.online.parameters())).double() - before_parameters
            close(after["linear_gap_change"], float(derivative @ delta), "independent branch linear projection")
    return models["actual"], stats["actual"]


def verify_q_arithmetic(row, initial=None):
    values = row["q_values"]
    require(len(values) == 8 and all(math.isfinite(value) for value in values), "eight finite probe Q values")
    action, competitor = row["demonstrated_action"], row["fixed_competitor"]
    require(isinstance(action, int) and isinstance(competitor, int)
            and 0 <= action < 8 and 0 <= competitor < 8 and action != competitor, "distinct valid reference and competitor actions")
    best_other = max((index for index in range(8) if index != action), key=lambda index: values[index])
    require(row["greedy_action"] == max(range(8), key=lambda index: values[index])
            and row["fixed_gap"] == values[action] - values[competitor], "Q-based greedy action and fixed gap")
    if initial is None:
        require(competitor == best_other, "initial competitor is the strongest alternative")
        return
    require(row["name"] == initial["name"] and action == initial["demonstrated_action"]
            and competitor == initial["fixed_competitor"] and row["best_other_after"] == best_other
            and row["competitor_changed"] == (best_other != competitor)
            and row["greedy_gap"] == values[action] - values[best_other], "fixed versus updated competitor semantics")
    close(row["observed_fixed_gap_change"], row["fixed_gap"] - initial["fixed_gap"], "actual fixed-gap change", rtol=0)
    close(row["linear_gap_prediction"], initial["fixed_gap"] + row["linear_gap_change"], "linear gap prediction", rtol=0)
    close(row["taylor_residual"], row["fixed_gap"] - row["linear_gap_prediction"], "reported nonlinear residual", rtol=0)


def expected_coefficients(weight):
    actual = {"td": 1.0, "margin": 1.0, "mc": weight}
    return {"actual": actual, "no_td": {**actual, "td": 0.0}, "no_margin": {**actual, "margin": 0.0},
            "no_mc": {**actual, "mc": 0.0}, "td_only": {"td": 1.0, "margin": 0.0, "mc": 0.0},
            "margin_only": {"td": 0.0, "margin": 1.0, "mc": 0.0},
            "mc_only": {"td": 0.0, "margin": 0.0, "mc": 1.0},
            "zero_current_gradient": {"td": 0.0, "margin": 0.0, "mc": 0.0},
            "zero_first_moment": actual, "reset_adam": actual, "no_clipping": actual}


def weighted_loss(values, coefficients):
    total = None
    for name in ("td", "margin", "mc"):
        if coefficients[name]:
            term = torch.tensor(values[name], dtype=torch.float32) * coefficients[name]
            total = term if total is None else total + term
    return 0.0 if total is None else float(total)


def verify_case_structure(record, metadata):
    """Audit every selected case, including branches not independently rerun."""
    finite_tree(record)
    update, weight = record["incoming_td_update"], float(record["arm"] == "mc_anchor")
    require(update in CASES and record["arm"] in ("baseline", "mc_anchor")
            and record["mc_weight"] == weight and record["source_td_update"] == record["pre_update_td"] == update - 1
            and record["margin"] == 0.8 and record["batch_size"] == 117,
            "declared case, objective and exact pre-update clock")
    require(record["source_state_and_rng_unchanged"] and record["exact_historical_log_and_sentinel"]
            and record["exact_original_updater_state"] and record["environment_actions"] == record["environment_resets"] == 0,
            "isolated diagnostic branch provenance and no physical actions")
    require([row["name"] for row in record["probes"]] == [row["name"] for row in metadata], "all nine declared probes")
    for row, meta in zip(record["probes"], metadata, strict=True):
        require(row["demonstrated_action"] == meta["demonstrated_action"], "unchanged probe reference label")
        verify_q_arithmetic(row)
        close(sum(row["adam_term_projections"].values()), row["adam_mathematical_delta_projection"], "additive Adam projections")
    specifications = expected_coefficients(weight)
    require(set(record["counterfactuals"]) == set(specifications), "eleven named branch slots")
    count = 0
    for name, coefficients in specifications.items():
        branch = record["counterfactuals"][name]
        if name == "no_mc" and weight == 0:
            require(branch["alias_of"] == "actual" and branch["optimizer_updates"] == 0
                    and set(branch) == {"alias_of", "optimizer_updates", "reason"}, "baseline no-MC is an unexecuted alias")
            continue
        count += 1
        require(branch["alias_of"] is None and branch["optimizer_updates"] == 1
                and branch["coefficients"] == coefficients and branch["source_td_update"] == update - 1
                and branch["result_td_update"] == update and branch["target_synchronized"] == (update % 200 == 0),
                "isolated one-step branch and absolute target clock")
        before_step = 0 if name == "reset_adam" else 20000 + update - 1
        after_step = 1 if name == "reset_adam" else 20000 + update
        require(branch["optimizer_steps_before"] == {"initialized_parameters": 0 if name == "reset_adam" else 6,
                                                      "minimum": before_step, "maximum": before_step}
                and branch["optimizer_steps_after"] == {"initialized_parameters": 6, "minimum": after_step, "maximum": after_step},
                "branch-specific Adam moments and bias-correction clocks")
        require(branch["losses_before"] == record["losses_before"]
                and branch["loss_before"] == weighted_loss(branch["losses_before"], coefficients)
                and branch["loss_after"] == weighted_loss(branch["losses_after_frozen_targets"], coefficients),
                "branch objective is the declared sum of means")
        expected_clip = 1.0 if name == "no_clipping" else float(torch.clamp(
            10.0 / (torch.tensor(branch["gradient_norm"], dtype=torch.float32) + 1e-6), max=1.0))
        require(branch["clip_coefficient"] == expected_clip and branch["clipping_enabled"] == (name != "no_clipping"),
                "one global clipping coefficient with the recorded original norm")
        require([row["name"] for row in branch["probes"]] == [row["name"] for row in metadata], "nine branch probes")
        for row, initial in zip(branch["probes"], record["probes"], strict=True):
            verify_q_arithmetic(row, initial)
            if name == "actual":
                require(row["linear_gap_change"] == initial["actual_delta_projection"], "actual branch shares displacement projection")
    require(record["independent_optimizer_steps"] == count == (11 if weight else 10), "executed versus aliased branch accounting")
    residual = record["adam_decomposition"]
    maximum, magnitude = residual["actual_minus_mathematical_max_abs"], residual["actual_minus_mathematical_norm"]
    norm_upper_bound = math.sqrt(record["parameter_elements"]) * maximum * (1 + torch.finfo(torch.float64).eps)
    require(0 <= maximum <= magnitude <= norm_upper_bound
            and residual["actual_delta_norm"] == record["counterfactuals"]["actual"]["parameter_delta_norm_float64"],
            "algebraically consistent recorded Adam residual norms")
    return count


def verify_protocol(protocol, settings):
    expected = {"attempt": 18, "kind": "retrospective_gradient_and_adam_diagnosis",
                "incoming_td_updates": list(CASES), "arms": ["baseline", "mc_anchor"],
                "effective_batch_size": 117, "margin": 0.8, "gamma": 0.99,
                "learning_rate": 0.0001, "gradient_clip_norm": 10.0,
                "baseline_mc_weight": 0.0, "treatment_mc_weight": 1.0,
                "environment_resets": 1, "environment_steps": 39, "completed_evaluation_episodes": 0,
                "new_policy_training_updates": 0, "new_demonstrations": 0, "held_out_scenes": 0}
    for name, value in expected.items():
        require(protocol[name] == value, f"fixed diagnostic protocol: {name}")
    require("Retrospective" in protocol["selection"] and "not a random sample" in protocol["selection"],
            "explicit retrospective case-selection limitation")
    require(settings["dqn"]["learning_rate"] == 0.0001 and settings["dqn"]["gamma"] == 0.99
            and settings["dqn"]["hidden_sizes"] == [256, 256]
            and settings["dqn"]["target_update_interval"] == 200
            and settings["dqn"]["gradient_clip_norm"] == 10, "unchanged historical network and update configuration")
    expected_scripts = {f"docs/experiments/018_gradient_diagnostics/{name}"
                        for name in ("run_gradient_diagnostics.py", "gradient_core.py", "trajectory_probe.py")}
    require(set(protocol["scripts_sha256"]) == expected_scripts, "complete main diagnostic script map")
    for name, digest in protocol["scripts_sha256"].items():
        require(file_hash(ROOT / name) == digest, f"unchanged diagnostic script: {name}")


def verify_probes(run_dir, reference, batch):
    """Load the frozen new observation; never replay or reset an environment."""
    metadata = read_json(run_dir / "probe_metadata.json")
    encoded = read_json(run_dir / "probe_observations.json")
    trajectory = read_json(run_dir / "trajectory_probe.json")
    finite_tree(trajectory)
    require(len(metadata) == 9 and len(encoded) == 9, "eight demonstration probes and one trajectory probe")
    probes = []
    for row in metadata:
        observation = torch.tensor(encoded[row["name"]], dtype=torch.float32)
        require(observation.shape == (1, 341) and bool(torch.isfinite(observation).all()), "finite encoded probe input")
        probes.append({"name": row["name"], "observation": observation,
                       "demonstrated_action": row["demonstrated_action"]})
    for row, probe, (seed, step, index) in zip(metadata[:8], probes[:8], REFERENCE_PROBES, strict=True):
        require(row == {"name": f"scene_{seed}_step_{step}", "scene_seed": seed, "reference_step": step,
                        "global_transition_index": index, "demonstrated_action": int(batch.actions[index])},
                "fixed reference probe identity")
        require_equal(probe["observation"], batch.observations[index:index + 1], "unchanged original reference observation")
    extra = metadata[-1]
    require(extra == trajectory["metadata"] and extra["name"] == "scene_10002_actual_step_40"
            and extra["scene_seed"] == 10002 and extra["reference_step"] == 40
            and extra["source_trajectory_checkpoint"] == 8500
            and extra["reference_action"] == extra["demonstrated_action"] == 1
            and extra["reference_action_kind"] == "successful_policy_action"
            and extra["reference_action_sources"] == [7500, 9000] and extra["fixed_comparison_action"] == 6,
            "new trajectory probe has an explicitly non-demonstration reference label")
    array = np.asarray(trajectory["encoded_observation"], dtype=np.float32)
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    require(array.shape == (341,) and extra["encoded_observation_shape"] == [1, 341]
            and digest == extra["encoded_observation_sha256"] == trajectory["encoded_observation_sha256"],
            "new trajectory probe exact float32 byte identity")
    require_equal(probes[-1]["observation"], torch.from_numpy(array).unsqueeze(0), "same persisted new probe tensor")
    directory = reference / "treatment/td"
    source_files = [directory / name for name in ("summary.json", "observations.json", "update_7500.pt", "update_8500.pt", "update_9000.pt")]
    require(trajectory["source_files_sha256"] == {str(path): file_hash(path) for path in source_files},
            "unchanged complete source evidence for the actual trajectory probe")
    observations = read_json(directory / "observations.json")
    rollouts = {update: next(scene["rollout"] for row in observations if row["update"] == update
                             for scene in row["scenes"] if scene["seed"] == 10002)
                for update in (7500, 8500, 9000)}
    shared = rollouts[8500]["actions"][:39]
    require(len(shared) == len(trajectory["prefix"]) == 39 and trajectory["shared_prefix_actions"] == shared,
            "exact 39-action common historical prefix")
    for update, rollout in rollouts.items():
        require(rollout["actions"][:39] == shared and rollout["success"] == (update != 8500)
                and rollout["actions"][39] == (6 if update == 8500 else 1), "historical shared prefix and step-40 fork")
        for index, row in enumerate(trajectory["prefix"]):
            require(row["step"] == index + 1 and row["action"] == shared[index]
                    and row["position_before"] == rollout["robot_path"][index]
                    and row["position_after"] == rollout["robot_path"][index + 1], "prefix actions and positions match saved history")
            values = row["newly_inferred_q_values"][str(update)]
            require(len(values) == 8 and int(np.argmax(values)) == shared[index], "newly inferred prefix Q values support logged actions")
    require(trajectory["accounting"] == {"environment_resets": 1, "environment_steps": 39, "completed_episodes": 0,
                                          "prefix_policy_forwards": 117, "step40_policy_forwards": 3,
                                          "optimizer_updates": 0, "new_demonstrations": 0}, "physical reconstruction costs")
    flags = trajectory["verification"]
    require(flags["all_logged_actions_exact"] and flags["all_logged_positions_exact"]
            and flags["three_policies_queried_on_one_common_state"]
            and flags["historical_per_step_q_available"] is False, "honest distinction between logged and newly inferred Q values")
    comparison = trajectory["demonstration_state_comparison"]
    original = batch.observations[115].numpy()
    difference = np.abs(array.astype(np.float64) - original.astype(np.float64))
    require(comparison == {"global_transition_index": 115, "scene_seed": 10002, "reference_step": 40,
                            "encoded_observation_sha256": hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest(),
                            "exactly_equal": bool(np.array_equal(array, original)),
                            "allclose_atol_1e_7_rtol_0": bool(np.allclose(array, original, atol=1e-7, rtol=0)),
                            "maximum_absolute_difference": float(difference.max()),
                            "different_feature_count": int(np.count_nonzero(difference))}, "new trajectory versus demonstration state comparison")
    reset = trajectory["reset_certification"]
    require(reset["original_verify_reference_route"] is True
            and reset["reconstruction_verify_reference_route"] is False
            and reset["initial_encoded_observation_exact"]
            and reset["original_initial_global_transition_index"] == 76,
            "only redundant reset certification disabled, original accepted scene preserved")
    expected_initial_hash = hashlib.sha256(np.ascontiguousarray(batch.observations[76].numpy()).tobytes()).hexdigest()
    require(reset["original_initial_encoded_observation_sha256"] == expected_initial_hash
            and reset["reconstructed_initial_encoded_observation_sha256"] == expected_initial_hash,
            "initial reconstructed state equals the original accepted demonstration scene")
    require([row["checkpoint_update"] for row in trajectory["step40_policies"]] == [7500, 8500, 9000],
            "all three source policy observations at actual step 40")
    with torch.random.fork_rng(devices=[]):
        for row in trajectory["step40_policies"]:
            update = row["checkpoint_update"]
            payload = torch.load(directory / f"update_{update}.pt", map_location="cpu", weights_only=True)
            model = build_q_network(341, 8, DQNConfig(**payload["dqn_config"]))
            model.load_state_dict(payload["online"])
            model.eval()
            with torch.no_grad():
                values = model(probes[-1]["observation"])[0].tolist()
            require(row["newly_inferred_q_values"] == values
                    and row["greedy_action"] == row["logged_action"] == int(np.argmax(values))
                    and row["south_small_minus_east_large"] == values[1] - values[6]
                    and row["historical_rollout_success"] == (update != 8500), "independent source-model Q values at the stored actual state")
    return probes, metadata


def verify(run_dir, output):
    """Audit all evidence and independently replay only the six declared cases."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite independent diagnostic verification: {output}")
    torch.set_num_threads(1)
    runner = load_module("verify018_runner_io", Path(__file__).with_name("run_gradient_diagnostics.py"))
    require(read_json(run_dir / "run_status.json")["status"] == "completed", "completed diagnostic run")
    settings, summary = read_json(run_dir / "settings.json"), read_json(run_dir / "summary.json")
    protocol = settings["training"]
    require(summary["protocol"] == protocol and summary["source_hashes_unchanged"], "matching protocol and immutable source assertion")
    verify_protocol(protocol, settings)
    manifest = read_json(run_dir / "input_manifest.json")
    reference = Path(manifest["reference_run"])
    print("Auditing historical source hashes and the original demonstration chain.", flush=True)
    inputs, arms, source_hashes = runner.audit_inputs(reference)
    require(source_hashes == manifest["files_sha256"], "complete independently rehashed source manifest")
    artifacts = summary["diagnostic_artifacts_sha256"]
    require(set(artifacts) == {"probe_metadata.json", "probe_observations.json", "trajectory_probe.json"},
            "complete probe artifact hash map")
    for name, digest in artifacts.items():
        require(file_hash(run_dir / name) == digest, f"unaltered stored probe artifact: {name}")
    probes, metadata = verify_probes(run_dir, reference, inputs["batch"])
    require(protocol["reference_probes"] == metadata[:8], "declared fixed demonstration probes")
    require(len(summary["cases"]) == 28, "all 28 selected historical cases")
    by_case, branch_count = {}, 0
    for entry in summary["cases"]:
        arm, update = entry["arm"], entry["incoming_td_update"]
        key = (arm, update)
        require(key not in by_case and arm in arms and update in CASES
                and entry["file"] == f"cases/{arm}_{update}.json", "unique declared case path")
        path = run_dir / entry["file"]
        require(file_hash(path) == entry["sha256"], "case evidence byte identity")
        record = read_json(path)
        require((record["arm"], record["incoming_td_update"]) == key, "case index agrees with record")
        branch_count += verify_case_structure(record, metadata)
        history = arms[arm]
        start = max(step for step in history["snapshots"] if step < update)
        source = history["snapshots"][start]
        groups = {step: [case for case in CASES if max(s for s in history["snapshots"] if s < case) == step]
                  for step in history["snapshots"]}
        expected_full = sorted(step for step in history["snapshots"] if start < step <= update)
        require(record["source_snapshot"] == str(source) and record["source_snapshot_sha256"] == file_hash(source)
                and record["restored_td_update"] == start and record["logical_prefix_length"] == update - start - 1
                and record["shared_replay_group"] == groups[start]
                and record["checked_full_state_updates"] == expected_full, "exact grouped restoration and full-state comparison coverage")
        by_case[key] = record
    require(set(by_case) == {(arm, update) for arm in arms for update in CASES}
            and {path.name for path in (run_dir / "cases").glob("*.json")}
            == {f"{arm}_{update}.json" for arm, update in by_case}, "complete case set with no extra hidden cases")
    accounting = read_json(run_dir / "accounting.json")
    expected_accounting = {"historical_replay_optimizer_updates": 2762,
                           "diagnostic_branch_optimizer_updates": 294,
                           "historical_post_update_verification_steps": 28, "completed_cases": 28,
                           "new_policy_training_updates": 0, "environment_resets": 1, "environment_steps": 39,
                           "completed_evaluation_episodes": 0}
    require(accounting == summary["accounting"] == expected_accounting and branch_count == 294,
            "all main historical-replay, branch, alias and physical-prefix accounting")
    selected_checks, replay_steps, branch_steps = [], 0, 0
    for arm, history in arms.items():
        for update in INDEPENDENT_CASES:
            print(f"Independently verifying {arm}, incoming TD {update}.", flush=True)
            start = max(step for step in history["snapshots"] if step < update)
            payload = torch.load(history["snapshots"][start], map_location="cpu", weights_only=True)
            agent = runner.restore_agent(payload, 341, 8)
            for step in range(start + 1, update):
                stats = runner.historical_update(agent, inputs, history)
                replay_steps += 1
                runner.compare_historical_step(agent, inputs, history, stats, step)
            before = runner.adam.training_snapshot(agent, payload["environment_config"])
            numerical_audit = {}
            actual, stats = verify_numeric_case(by_case[(arm, update)], agent.learner,
                                               inputs["batch"], inputs["returns"], probes, numerical_audit)
            branch_steps += 3
            require_equal(runner.adam.training_snapshot(agent, payload["environment_config"]), before,
                          "independent diagnostics preserve original model, Adam and both RNGs")
            agent.learner = actual
            if not history["weight"]:
                stats = {name: value for name, value in stats.items() if name not in anchor.EXTRA_FIELDS}
            require(runner.compare_historical_step(agent, inputs, history, stats, update),
                    "each selected actual post-state has an exact saved full-state counterpart")
            selected_checks.append({"arm": arm, "incoming_td_update": update, "restored_td_update": start,
                                    "preceding_historical_replay_updates": update - start - 1,
                                    "independent_optimizer_updates": 3, "probe_count": len(probes),
                                    "exact_original_log_sentinel_and_full_snapshot": True,
                                    "independent_component_gradients_adam_parts_and_probe_derivatives": True,
                                    "exact_actual_reset_adam_and_zero_first_moment_branches": True,
                                    **numerical_audit})
    require(replay_steps == 198 and branch_steps == 18, "strict independent verification update budget")
    for name, digest in source_hashes.items():
        require(file_hash(Path(name)) == digest, f"verification leaves historical source immutable: {name}")
    result = {"passed": True, "run_dir": str(run_dir), "audited_case_files": 28,
              "probes_per_case": 9, "numerically_verified_cases": selected_checks,
              "main_accounting": accounting,
              "verification_accounting": {"preceding_historical_replay_optimizer_updates": replay_steps,
                                           "independent_actual_and_counterfactual_optimizer_updates": branch_steps,
                                           "total_additional_optimizer_updates": replay_steps + branch_steps,
                                           "environment_resets": 0, "environment_steps": 0,
                                           "completed_evaluation_episodes": 0,
                                           "source_policy_new_probe_forward_passes": 3},
              "source_hashes_unchanged": True, "verifier_sha256": file_hash(Path(__file__)),
              "scope": "All 28 records audited; independent numerical reconstruction only at incoming TD 2008, 2201 and 8500 in each arm. No new physical trajectory replay or policy success evaluation.",
              "trajectory_probe_verification": "Stored input bytes, source models, recorded prefix actions and positions, and three new-state Q vectors checked. Prefix physics and per-prefix Q vectors are not independently recomputed here."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Independent diagnostics verified: six numerical cases, 216 optimizer steps, no environment actions.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
