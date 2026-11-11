"""Controlled Adam-state transitions from calibrated action values to TD learning."""

import hashlib
import json
from copy import deepcopy
from math import isfinite, sqrt

import torch

from me5418_harvesting.agents.dqn_learning import DQNUpdater


def _canonical(value):
    """Represent optimizer settings and tensors without serialization metadata."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Optimizer tensors must contain finite values.")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "device": str(value.device), "data_sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError(f"Unsupported or non-finite optimizer setting: {type(value).__name__}")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parameter_names(learner):
    if type(learner.optimizer) is not torch.optim.Adam:
        raise ValueError("The controlled transition requires a standard Adam optimizer.")
    parameters = dict(learner.online.named_parameters())
    names = {id(parameter): name for name, parameter in parameters.items()}
    attached = [parameter for group in learner.optimizer.param_groups for parameter in group["params"]]
    if (not parameters or len(attached) != len(names)
            or {id(parameter) for parameter in attached} != set(names)):
        raise ValueError("Adam must contain every online parameter exactly once.")
    if any(id(parameter) not in names for parameter in learner.optimizer.state):
        raise ValueError("Adam contains state for an unattached parameter.")
    return parameters, names


def optimizer_state_summary(learner: DQNUpdater) -> dict:
    """Hash named Adam state and settings, separately from the TD update clock.

    Tensor bytes, dtype, shape and device enter the hash. Parameter names replace
    process-specific identities. No model weights or zip timestamps enter it.
    """
    parameters, names = _parameter_names(learner)
    optimizer = learner.optimizer
    groups = [
        {**_canonical({key: value for key, value in group.items() if key != "params"}),
         "parameters": [names[id(parameter)] for parameter in group["params"]]}
        for group in optimizer.param_groups
    ]
    settings = {"defaults": _canonical(optimizer.defaults), "parameter_groups": groups}
    states, steps = {}, {}
    moment_squares = {"exp_avg": 0.0, "exp_avg_sq": 0.0}
    for name, parameter in parameters.items():
        state = optimizer.state.get(parameter, {})
        if not state:
            continue
        states[name] = _canonical(state)
        if "step" in state:
            steps[name] = float(state["step"])
        for key in moment_squares:
            if key in state:
                moment_squares[key] += state[key].detach().double().square().sum().item()
    return {
        "sha256": _digest({"settings": settings, "states": states}),
        "settings_sha256": _digest(settings), "parameter_groups": groups,
        "parameter_count": len(parameters), "initialized_parameter_count": len(states),
        "steps": steps, "step_min": min(steps.values()) if steps else None,
        "step_max": max(steps.values()) if steps else None,
        "first_moment_l2": sqrt(moment_squares["exp_avg"]),
        "second_moment_l2": sqrt(moment_squares["exp_avg_sq"]),
    }


def _validate_transition(learner, preserve_optimizer):
    if not isinstance(preserve_optimizer, bool):
        raise TypeError("preserve_optimizer must be a boolean.")
    parameters, _ = _parameter_names(learner)
    if (isinstance(learner.update_count, bool) or not isinstance(learner.update_count, int)
            or learner.update_count < 0):
        raise ValueError("The previous phase update count must be a nonnegative integer.")
    online_state, target_state = learner.online.state_dict(), learner.target.state_dict()
    if online_state.keys() != target_state.keys() or any(
        value.shape != target_state[name].shape for name, value in online_state.items()
    ):
        raise ValueError("Online and target network layouts must match before transition.")
    if any(not torch.isfinite(value).all() for value in online_state.values()):
        raise ValueError("Online network values must be finite before transition.")
    for group in learner.optimizer.param_groups:
        for parameter in group["params"]:
            state = learner.optimizer.state.get(parameter, {})
            if not state:
                if preserve_optimizer:
                    raise ValueError("Preserving Adam requires initialized state for every parameter.")
                continue
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad", False):
                required.add("max_exp_avg_sq")
            if not required.issubset(state):
                raise ValueError("Adam state is incomplete for an online parameter.")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or not torch.isfinite(step).all() or step.item() < 1
                    or step.item() != int(step.item())):
                raise ValueError("Initialized Adam steps must be finite positive integers.")
            for key in required - {"step"}:
                value = state[key]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or value.device != parameter.device
                        or not torch.isfinite(value).all()):
                    raise ValueError("Adam moments must match their finite online parameters.")
                if key != "exp_avg" and bool((value < 0).any()):
                    raise ValueError("Adam second moments must be nonnegative.")
    if not parameters:
        raise ValueError("The online network has no parameters.")


def begin_td_with_optimizer(learner: DQNUpdater, *, preserve_optimizer: bool) -> dict:
    """Synchronize TD targets while either retaining or resetting all Adam state.

    Retention keeps first and second moments and per-parameter bias-correction
    steps. Both arms reset the learner's target-synchronization clock to zero.
    Resetting preserves optimizer settings, including nondefault group options.
    Validation and replacement construction precede any learner mutation.
    """
    _validate_transition(learner, preserve_optimizer)
    before = optimizer_state_summary(learner)
    replacement = learner.optimizer
    if not preserve_optimizer:
        groups = [
            {key: list(value) if key == "params" else deepcopy(value)
             for key, value in group.items()}
            for group in learner.optimizer.param_groups
        ]
        replacement = torch.optim.Adam(groups, **deepcopy(learner.optimizer.defaults))
    previous_updates = learner.update_count
    learner.optimizer = replacement
    learner.online.zero_grad(set_to_none=True)
    learner.target.load_state_dict(learner.online.state_dict())
    learner.target.requires_grad_(False)
    learner.target.eval()
    learner.target.zero_grad(set_to_none=True)
    learner.update_count = 0
    after = optimizer_state_summary(learner)
    return {
        "previous_phase_updates": previous_updates, "phase_updates": 0,
        "optimizer_reset": not preserve_optimizer, "target_synchronized": True,
        "optimizer_before": before, "optimizer_after": after,
    }
