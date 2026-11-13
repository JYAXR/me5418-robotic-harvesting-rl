"""Inspect loss gradients and independent one-step Adam counterfactuals without training."""

from copy import deepcopy
from itertools import combinations
from math import isfinite, sqrt
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from me5418_harvesting.agents.dqn_learning import demonstration_margin_loss, td_loss

COMPONENTS = ("td", "margin", "mc")
MARGIN = 0.8
BRANCH_NAMES = (
    "actual", "no_td", "no_margin", "no_mc", "td_only", "margin_only", "mc_only",
    "zero_current_gradient", "zero_first_moment", "reset_adam", "no_clipping",
)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _source_state(learner):
    return deepcopy({
        "online": learner.online.state_dict(), "target": learner.target.state_dict(),
        "optimizer": learner.optimizer.state_dict(), "update_count": learner.update_count,
        "online_modes": [module.training for module in learner.online.modules()],
        "target_modes": [module.training for module in learner.target.modules()],
        "gradients": [parameter.grad for parameter in learner.online.parameters()],
    })


def _rng_state():
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def _restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _flat(tensors):
    return torch.cat([tensor.detach().reshape(-1).double().cpu() for tensor in tensors])


def _norm(vector):
    return float(torch.linalg.vector_norm(vector))


def _dot(left, right):
    return float(torch.dot(left, right))


def _layout(model):
    parameters, offset = [], 0
    layers = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        size = parameter.numel()
        layer = name.rsplit(".", 1)[0] if "." in name else "root"
        record = {"name": name, "shape": list(parameter.shape), "start": offset,
                  "stop": offset + size, "layer": layer}
        parameters.append(record)
        layers.setdefault(layer, []).append(record)
        offset += size
    return parameters, layers


def _subset(vector, pieces):
    return torch.cat([vector[row["start"]:row["stop"]] for row in pieces])


def _alignment(vectors):
    norms = {name: _norm(value) for name, value in vectors.items()}
    pairs = []
    for first, second in combinations(vectors, 2):
        product = _dot(vectors[first], vectors[second])
        denominator = norms[first] * norms[second]
        pairs.append({"first": first, "second": second, "dot": product,
                      "cosine": product / denominator if denominator else None})
    return {"norms": norms, "pairs": pairs}


def _validate(learner, batch, returns, probes, mc_weight):
    if isinstance(mc_weight, bool) or not isinstance(mc_weight, Real) or not isfinite(mc_weight) or mc_weight not in (0, 1):
        raise ValueError("The diagnostic requires declared MC weight 0 or 1.")
    if (not isinstance(returns, torch.Tensor) or returns.ndim != 1 or not returns.numel()
            or returns.shape != batch.rewards.shape or returns.dtype != batch.observations.dtype
            or returns.device != batch.observations.device or not returns.is_floating_point()
            or not torch.isfinite(returns).all()):
        raise ValueError("Fixed returns must be finite and match the observation layout and reward vector.")
    if (batch.observations.ndim != 2 or len(batch.observations) != len(returns)
            or batch.actions.shape != returns.shape or batch.actions.dtype != torch.long):
        raise ValueError("Observations and demonstration actions must match the return targets.")
    if not isinstance(learner.optimizer, torch.optim.Adam):
        raise TypeError("The decomposition supports the project's ordinary Adam optimizer only.")
    online = [parameter for parameter in learner.online.parameters() if parameter.requires_grad]
    optimized = [parameter for group in learner.optimizer.param_groups for parameter in group["params"]]
    if not online or len(online) != len(optimized) or {id(p) for p in online} != {id(p) for p in optimized}:
        raise ValueError("Adam must own every trainable online parameter exactly once.")
    for group in learner.optimizer.param_groups:
        if (group.get("weight_decay", 0) != 0 or group.get("amsgrad", False) or group.get("maximize", False)
                or group.get("capturable", False) or group.get("differentiable", False)):
            raise ValueError("Unsupported Adam variant: weight decay, AMSGrad, maximize, capturable or differentiable mode.")
    names = []
    for probe in probes:
        observation = probe["observation"]
        if (not isinstance(observation, torch.Tensor) or observation.shape != (1, batch.observations.shape[1])
                or observation.dtype != batch.observations.dtype or observation.device != batch.observations.device
                or not torch.isfinite(observation).all()):
            raise ValueError("Each probe must be one finite observation with the training layout.")
        if not isinstance(probe["name"], str) or not probe["name"]:
            raise ValueError("Every probe needs a unique nonempty name.")
        action = probe["demonstrated_action"]
        if isinstance(action, bool) or not isinstance(action, int) or action < 0:
            raise ValueError("The demonstrated probe action must be a nonnegative integer.")
        names.append(probe["name"])
    if not names or len(names) != len(set(names)):
        raise ValueError("The diagnostic requires distinct named probes.")


def _losses(learner, batch, returns):
    # This graph order matches production TD and attempt-017 MC anchoring.
    td, chosen, targets = td_loss(learner.online, learner.target, batch, learner.config.gamma)
    values = learner.online(batch.observations)
    margin = demonstration_margin_loss(values, batch.actions, MARGIN)
    mc = F.smooth_l1_loss(chosen, returns.detach())
    return {"td": td, "margin": margin, "mc": mc}, targets, values


def _objective(losses, coefficients):
    total = None
    for name in COMPONENTS:
        weight = coefficients[name]
        if not weight:
            continue
        term = losses[name] if name == "td" and weight == 1 else weight * losses[name]
        total = term if total is None else total + term
    return losses["td"].new_zeros(()) if total is None else total


def _after_losses(learner, batch, returns, targets):
    with torch.no_grad():
        values = learner.online(batch.observations)
        chosen = values.gather(1, batch.actions[:, None]).squeeze(1)
        return {"td": F.smooth_l1_loss(chosen, targets),
                "margin": demonstration_margin_loss(values, batch.actions, MARGIN),
                "mc": F.smooth_l1_loss(chosen, returns.detach())}


def _component_gradients(learner, batch, returns, coefficients):
    clone = deepcopy(learner)
    clone.online.train()
    clone.optimizer.zero_grad(set_to_none=True)
    parameters = [parameter for parameter in clone.online.parameters() if parameter.requires_grad]
    losses, targets, _ = _losses(clone, batch, returns)
    vectors = {}
    for name in COMPONENTS:
        gradients = torch.autograd.grad(losses[name], parameters, retain_graph=True)
        vectors[name] = _flat(gradients)
    actual = _objective(losses, coefficients)
    actual.backward()
    vectors["actual"] = _flat([parameter.grad for parameter in parameters])
    if not all(torch.isfinite(vector).all() for vector in vectors.values()):
        raise FloatingPointError("All objective gradients must be finite.")
    weighted_sum = sum(coefficients[name] * vectors[name] for name in COMPONENTS)
    return vectors, weighted_sum, {name: float(value.detach()) for name, value in losses.items()}, targets.detach().clone()


def _probe_before(learner, probes):
    model = deepcopy(learner.online)
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    records, gradients = [], {}
    for probe in probes:
        values = model(probe["observation"].detach())[0]
        action = probe["demonstrated_action"]
        if values.ndim != 1 or len(values) < 2 or action >= len(values) or not torch.isfinite(values).all():
            raise ValueError("Probe Q values must be finite and include the demonstrated action and alternatives.")
        candidates = values.detach().clone()
        candidates[action] = -torch.inf
        competitor = int(candidates.argmax())
        gradient = torch.autograd.grad(values[action] - values[competitor], parameters)
        gradients[probe["name"]] = _flat(gradient)
        if not torch.isfinite(gradients[probe["name"]]).all():
            raise FloatingPointError("Probe gap gradients must be finite.")
        records.append({"name": probe["name"], "demonstrated_action": action,
                        "fixed_competitor": competitor, "greedy_action": int(values.detach().argmax()),
                        "q_values": values.detach().cpu().tolist(),
                        "fixed_gap": float(values[action].detach()) - float(values[competitor].detach()),
                        "gap_gradient_norm": _norm(gradients[probe["name"]])})
    return records, gradients


def _steps(optimizer):
    result = [float(state["step"]) for state in optimizer.state.values() if "step" in state]
    return {"initialized_parameters": len(result), "minimum": min(result) if result else 0.0,
            "maximum": max(result) if result else 0.0}


def _branch_specifications(mc_weight):
    ordinary = {"td": 1.0, "margin": 1.0, "mc": float(mc_weight)}
    return {
        "actual": ordinary,
        "no_td": {**ordinary, "td": 0.0}, "no_margin": {**ordinary, "margin": 0.0},
        "no_mc": {**ordinary, "mc": 0.0},
        "td_only": {"td": 1.0, "margin": 0.0, "mc": 0.0},
        "margin_only": {"td": 0.0, "margin": 1.0, "mc": 0.0},
        "mc_only": {"td": 0.0, "margin": 0.0, "mc": 1.0},
        "zero_current_gradient": {"td": 0.0, "margin": 0.0, "mc": 0.0},
        "zero_first_moment": ordinary, "reset_adam": ordinary, "no_clipping": ordinary,
    }


def _execute_branch(source, batch, returns, name, coefficients, expected_targets):
    learner = deepcopy(source)
    learner.online.train()
    if name == "zero_first_moment":
        for state in learner.optimizer.state.values():
            if "exp_avg" in state:
                state["exp_avg"].zero_()
    elif name == "reset_adam":
        learner.optimizer.state.clear()
    steps_before = _steps(learner.optimizer)
    learner.optimizer.zero_grad(set_to_none=True)
    parameters = [parameter for parameter in learner.online.parameters() if parameter.requires_grad]
    before_native = nn.utils.parameters_to_vector(parameters).detach().clone()
    losses, targets, values = _losses(learner, batch, returns)
    if not torch.equal(targets, expected_targets):
        raise RuntimeError("Independent branches must use identical frozen Bellman targets.")
    objective = _objective(losses, coefficients)
    if not torch.isfinite(objective):
        raise FloatingPointError("A counterfactual objective is not finite.")
    if name == "zero_current_gradient":
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
    else:
        objective.backward()
    raw_gradients = [parameter.grad for parameter in parameters]
    raw = _flat(raw_gradients)
    if not torch.isfinite(raw).all():
        raise FloatingPointError("A counterfactual gradient is not finite.")
    if name == "no_clipping":
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in raw_gradients]))
        clip_coefficient = 1.0
    else:
        norm = nn.utils.clip_grad_norm_(parameters, learner.config.gradient_clip_norm, error_if_nonfinite=True)
        clip_coefficient = float(torch.clamp(learner.config.gradient_clip_norm / (norm + 1e-6), max=1.0))
    clipped = _flat([parameter.grad for parameter in parameters])
    learner.optimizer.step()
    after_losses = _after_losses(learner, batch, returns, targets)
    after_native = nn.utils.parameters_to_vector(parameters).detach()
    delta = after_native.double().cpu() - before_native.double().cpu()
    if not torch.isfinite(delta).all() or not all(torch.isfinite(value) for value in after_losses.values()):
        raise FloatingPointError("An Adam counterfactual produced non-finite parameters or losses.")
    learner.update_count += 1
    synchronized = learner.update_count % learner.config.target_update_interval == 0
    if synchronized:
        learner.target.load_state_dict(learner.online.state_dict())
    record = {
        "alias_of": None, "optimizer_updates": 1, "coefficients": dict(coefficients),
        "source_td_update": source.update_count, "result_td_update": learner.update_count,
        "target_synchronized": synchronized, "optimizer_steps_before": steps_before,
        "optimizer_steps_after": _steps(learner.optimizer),
        "losses_before": {key: float(value.detach()) for key, value in losses.items()},
        "losses_after_frozen_targets": {key: float(value.detach()) for key, value in after_losses.items()},
        "loss_before": float(objective.detach()), "loss_after": float(_objective(after_losses, coefficients).detach()),
        "demonstration_action_accuracy_before": float((values.detach().argmax(dim=1) == batch.actions).float().mean()),
        "gradient_norm": float(norm), "clip_coefficient": clip_coefficient,
        "clipping_enabled": name != "no_clipping", "raw_gradient_norm_float64": _norm(raw),
        "clipped_gradient_norm_float64": _norm(clipped),
        "parameter_delta_norm": float(torch.linalg.vector_norm(after_native - before_native)),
        "parameter_delta_norm_float64": _norm(delta),
    }
    return learner, record, delta, raw, clipped


def _probe_after(learner, probes, before, gradients, delta):
    model = learner.online
    modes = [(module, module.training) for module in model.modules()]
    records = []
    try:
        model.eval()
        with torch.no_grad():
            for probe, initial in zip(probes, before, strict=True):
                values = model(probe["observation"].detach())[0]
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Counterfactual probe Q values must be finite.")
                action, competitor = initial["demonstrated_action"], initial["fixed_competitor"]
                others = values.clone()
                others[action] = -torch.inf
                next_competitor = int(others.argmax())
                fixed_gap = float(values[action]) - float(values[competitor])
                greedy_gap = float(values[action]) - float(values[next_competitor])
                projection = _dot(gradients[probe["name"]], delta)
                prediction = initial["fixed_gap"] + projection
                records.append({"name": probe["name"], "q_values": values.cpu().tolist(),
                                "demonstrated_action": action, "greedy_action": int(values.argmax()),
                                "fixed_competitor": competitor, "best_other_after": next_competitor,
                                "competitor_changed": next_competitor != competitor,
                                "fixed_gap": fixed_gap, "greedy_gap": greedy_gap,
                                "observed_fixed_gap_change": fixed_gap - initial["fixed_gap"],
                                "linear_gap_change": projection, "linear_gap_prediction": prediction,
                                "taylor_residual": fixed_gap - prediction})
    finally:
        for module, mode in modes:
            module.training = mode
    return records


def _adam_decomposition(source, actual, component_vectors, weights, raw, clipped, delta, clip_coefficient, layout):
    old_parameters = dict(source.online.named_parameters())
    new_parameters = dict(actual.online.named_parameters())
    groups = {id(parameter): group for group in actual.optimizer.param_groups for parameter in group["params"]}
    pieces = {name: [] for name in ("old_first_moment", "current_td", "current_margin", "current_mc", "rounding_remainder")}
    parameter_records = []
    weighted_sum = sum(weights[name] * component_vectors[name] for name in COMPONENTS)
    for row in layout:
        previous, parameter = old_parameters[row["name"]], new_parameters[row["name"]]
        old_state, state = source.optimizer.state.get(previous, {}), actual.optimizer.state[parameter]
        group = groups[id(parameter)]
        beta1, beta2 = group["betas"]
        step = float(state["step"])
        old_m = old_state.get("exp_avg", torch.zeros_like(previous)).detach().reshape(-1).double().cpu()
        actual_m = state["exp_avg"].detach().reshape(-1).double().cpu()
        actual_v = state["exp_avg_sq"].detach().reshape(-1).double().cpu()
        denominator = actual_v.sqrt() / sqrt(1.0 - beta2 ** step) + group["eps"]
        factor = -float(group["lr"]) / (1.0 - beta1 ** step) / denominator
        terms = {"old_first_moment": beta1 * old_m}
        for name in COMPONENTS:
            gradient = component_vectors[name][row["start"]:row["stop"]]
            terms[f"current_{name}"] = (1.0 - beta1) * clip_coefficient * weights[name] * gradient
        terms["rounding_remainder"] = actual_m - sum(terms.values())
        for name, moment in terms.items():
            pieces[name].append(factor * moment)
        parameter_records.append({"name": row["name"], "step": step,
                                  "beta1": beta1, "beta2": beta2, "learning_rate": float(group["lr"]),
                                  "epsilon": group["eps"], "denominator_min": float(denominator.min()),
                                  "denominator_max": float(denominator.max())})
    vectors = {name: torch.cat(values) for name, values in pieces.items()}
    mathematical_delta = sum(vectors.values())
    remainder = delta - mathematical_delta
    record = {
        "common_clip_coefficient": clip_coefficient,
        "common_denominator": "Actual post-update second moment, with the actual step's bias correction and epsilon.",
        "terms": {name: {"norm": _norm(value)} for name, value in vectors.items()},
        "gradient_sum_residual_norm": _norm(raw - weighted_sum),
        "clip_rounding_residual_norm": _norm(clipped - clip_coefficient * raw),
        "mathematical_delta_norm": _norm(mathematical_delta), "actual_delta_norm": _norm(delta),
        "actual_minus_mathematical_norm": _norm(remainder),
        "actual_minus_mathematical_max_abs": float(remainder.abs().max()),
        "parameter_details": parameter_records,
        "rounding_remainder_semantics": "The first-moment remainder includes component accumulation, clipping and Adam moment rounding; the final delta residual also includes parameter storage and optimizer arithmetic rounding.",
        "interpretation": "These additive terms share the actual denominator. They are not counterfactual steps with separately recomputed second moments.",
    }
    return record, vectors, mathematical_delta


def diagnose_update(learner, batch, returns, probes, *, mc_weight):
    """Return JSON diagnostics and the actual learner copy; leave the source and RNG untouched."""
    probes = list(probes)
    _validate(learner, batch, returns, probes, mc_weight)
    fixed_returns = returns.detach()
    source_before, rng = _source_state(learner), _rng_state()
    actual_copy = None
    try:
        layout, layers = _layout(learner.online)
        weights = {"td": 1.0, "margin": 1.0, "mc": float(mc_weight)}
        components, weighted_sum, losses, targets = _component_gradients(learner, batch, fixed_returns, weights)
        before, gap_gradients = _probe_before(learner, probes)
        gradient_statistics = {
            "global": _alignment({name: components[name] for name in COMPONENTS}),
            "parameters": [{"name": row["name"], "shape": row["shape"],
                            **_alignment({name: _subset(components[name], [row]) for name in COMPONENTS})} for row in layout],
            "layers": [{"name": name,
                        **_alignment({component: _subset(components[component], rows) for component in COMPONENTS})}
                       for name, rows in layers.items()],
            "actual_gradient_norm": _norm(components["actual"]), "weighted_component_sum_norm": _norm(weighted_sum),
            "actual_minus_component_sum_norm": _norm(components["actual"] - weighted_sum),
            "zero_norm_cosine": "A cosine is null when either vector has zero norm.",
        }
        branches, displacements, actual_raw, actual_clipped = {}, {}, None, None
        for name, coefficients in _branch_specifications(mc_weight).items():
            if name == "no_mc" and mc_weight == 0:
                branches[name] = {"alias_of": "actual", "optimizer_updates": 0,
                                  "reason": "MC weight is already zero in the actual objective."}
                continue
            _restore_rng(rng)
            clone, record, delta, raw, clipped = _execute_branch(learner, batch, fixed_returns, name, coefficients, targets)
            record["probes"] = _probe_after(clone, probes, before, gap_gradients, delta)
            branches[name], displacements[name] = record, delta
            if name == "actual":
                actual_copy, actual_raw, actual_clipped = clone, raw, clipped
        if not torch.equal(components["actual"], actual_raw):
            raise RuntimeError("The measured actual gradient differs from the independently executed actual branch.")
        decomposition, adam_vectors, mathematical_delta = _adam_decomposition(
            learner, actual_copy, components, weights, actual_raw, actual_clipped,
            displacements["actual"], branches["actual"]["clip_coefficient"], layout,
        )
        for probe in before:
            gap = gap_gradients[probe["name"]]
            probe["negative_component_gradient_projections"] = {name: -_dot(gap, components[name]) for name in COMPONENTS}
            probe["weighted_negative_component_gradient_projections"] = {
                name: -weights[name] * _dot(gap, components[name]) for name in COMPONENTS}
            probe["negative_actual_gradient_projection"] = -_dot(gap, actual_raw)
            probe["negative_clipped_gradient_projection"] = -_dot(gap, actual_clipped)
            probe["parameter_projections"] = [{"name": row["name"],
                                               "gap_gradient_norm": _norm(_subset(gap, [row])),
                                               **{name: -_dot(_subset(gap, [row]), _subset(components[name], [row]))
                                                  for name in (*COMPONENTS, "actual")}} for row in layout]
            probe["layer_projections"] = [{"name": name, "gap_gradient_norm": _norm(_subset(gap, rows)),
                                           **{component: -_dot(_subset(gap, rows), _subset(components[component], rows))
                                              for component in (*COMPONENTS, "actual")}} for name, rows in layers.items()]
            probe["adam_term_projections"] = {name: _dot(gap, vector) for name, vector in adam_vectors.items()}
            probe["adam_mathematical_delta_projection"] = _dot(gap, mathematical_delta)
            probe["actual_delta_projection"] = _dot(gap, displacements["actual"])
        record = {
            "source_td_update": learner.update_count, "incoming_td_update": learner.update_count + 1,
            "mc_weight": float(mc_weight), "margin": MARGIN, "batch_size": len(batch.actions),
            "parameter_count": len(layout), "parameter_elements": layout[-1]["stop"],
            "losses_before": losses, "component_gradients": gradient_statistics,
            "probes": before, "adam_decomposition": decomposition, "counterfactuals": branches,
            "independent_optimizer_steps": sum(branch["optimizer_updates"] for branch in branches.values()),
            "environment_actions": 0, "environment_resets": 0,
            "projection_semantics": "Positive projections predict an increase of the fixed-before-competitor demonstration gap; nonlinear residuals and after-step competitor changes are separate.",
            "counterfactual_semantics": "Each non-alias branch performs one independent update from the same source state. Zero-current-gradient uses explicit zeros; zero-first-moment preserves second moments and steps; reset-Adam clears state only.",
        }
    finally:
        _restore_rng(rng)
        if not _equal(_source_state(learner), source_before) or not _equal(_rng_state(), rng):
            raise RuntimeError("Gradient diagnostics mutated source state or global RNG.")
    record["source_state_and_rng_unchanged"] = True
    return record, actual_copy
