"""Summarize retained local gradient diagnostics without inference or optimization."""

import argparse
import hashlib
import json
import math
from pathlib import Path

COMPONENTS = ("td", "margin", "mc")
BRANCHES = ("actual", "no_td", "no_margin", "no_mc", "td_only", "margin_only", "mc_only",
            "zero_current_gradient", "zero_first_moment", "reset_adam", "no_clipping")
TERMS = ("old_first_moment", "current_td", "current_margin", "current_mc", "rounding_remainder")
MAX_BYTES = 8 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Diagnostic records must not contain non-finite numbers.")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def read_json(path):
    require(path.stat().st_size <= MAX_BYTES, f"Diagnostic input exceeds the 8 MiB cap: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    finite_tree(value)
    return value


def resolve_branches(case):
    """Resolve named aliases without treating them as additional optimizer steps."""
    branches = case["counterfactuals"]
    require(set(branches) == set(BRANCHES), "The eleven declared diagnostic branches must be present.")
    resolved = {}
    for name in BRANCHES:
        visited, current = set(), name
        while branches[current].get("alias_of") is not None:
            require(current not in visited, "Counterfactual aliases must not contain cycles.")
            visited.add(current)
            current = branches[current]["alias_of"]
            require(current in branches, "A counterfactual alias names a missing branch.")
        resolved[name] = branches[current]
    return resolved


def sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def summarize_case(case, metadata):
    """Keep every selected state and branch; distinguish ranking and fixed-pair gaps."""
    finite_tree(case)
    require(case["source_state_and_rng_unchanged"] and case["exact_historical_log_and_sentinel"]
            and case["exact_original_updater_state"], "The historical reproduction gates must pass.")
    require(case["source_td_update"] == case["pre_update_td"] == case["incoming_td_update"] - 1,
            "Every incoming update must use the immediately preceding source state.")
    resolved = resolve_branches(case)
    before_rows = {row["name"]: row for row in case["probes"]}
    require(len(before_rows) == len(case["probes"]) and set(before_rows) == set(metadata),
            "Every declared probe must appear exactly once in each case.")
    after = {}
    for branch, record in resolved.items():
        after[branch] = {row["name"]: row for row in record["probes"]}
        require(len(after[branch]) == len(record["probes"]) and set(after[branch]) == set(metadata),
                "Each independent branch must retain all declared probes.")
    probes = []
    for name, before in before_rows.items():
        label = before["demonstrated_action"]
        require(label == metadata[name]["demonstrated_action"], "The probe's reference action changed.")
        actual = after["actual"][name]
        change = actual["fixed_gap"] - before["fixed_gap"]
        require(change == actual["observed_fixed_gap_change"], "The retained fixed-gap difference is inconsistent.")
        terms = before["adam_term_projections"]
        require(set(terms) == set(TERMS), "All common-denominator Adam projection terms are required.")
        storage = before["actual_delta_projection"] - before["adam_mathematical_delta_projection"]
        nonlinear = change - before["actual_delta_projection"]
        require(math.isclose(nonlinear, actual["taylor_residual"], rel_tol=0, abs_tol=1e-12),
                "The retained nonlinear gap residual is inconsistent.")
        branches = {}
        for branch, values in after.items():
            row = values[name]
            require(row["demonstrated_action"] == label
                    and row["fixed_competitor"] == before["fixed_competitor"],
                    "Counterfactuals must preserve the reference action and fixed pre-update competitor.")
            before_correct, after_correct = before["greedy_action"] == label, row["greedy_action"] == label
            branches[branch] = {
                "alias_of": case["counterfactuals"][branch].get("alias_of"),
                "after_greedy_gap": row["greedy_gap"], "after_fixed_gap": row["fixed_gap"],
                "fixed_gap_change": row["observed_fixed_gap_change"],
                "greedy_action": row["greedy_action"], "best_other_after": row["best_other_after"],
                "competitor_changed": row["competitor_changed"],
                "negative_to_positive_gap": before["fixed_gap"] < 0 < row["greedy_gap"],
                "positive_to_negative_gap": before["fixed_gap"] > 0 > row["greedy_gap"],
                "action_recovered_from_before": not before_correct and after_correct,
                "action_regressed_from_before": before_correct and not after_correct,
                "greedy_gap_difference_from_actual": row["greedy_gap"] - actual["greedy_gap"],
                "action_recovered_relative_to_actual": actual["greedy_action"] != label and after_correct,
                "action_regressed_relative_to_actual": actual["greedy_action"] == label and not after_correct,
            }
        naive = before["negative_actual_gradient_projection"]
        probes.append({
            "name": name, "reference_action": label,
            "reference_action_kind": metadata[name].get("reference_action_kind", "original_demonstration_action"),
            "fixed_competitor": before["fixed_competitor"], "greedy_action_before": before["greedy_action"],
            "before_gap": before["fixed_gap"], "actual_after_greedy_gap": actual["greedy_gap"],
            "actual_after_fixed_gap": actual["fixed_gap"], "actual_fixed_gap_change": change,
            "actual_greedy_gap_change": actual["greedy_gap"] - before["fixed_gap"],
            "actual_competitor_changed": actual["competitor_changed"],
            "negative_component_gradient_projections": before["negative_component_gradient_projections"],
            "weighted_negative_component_gradient_projections": before["weighted_negative_component_gradient_projections"],
            "negative_actual_gradient_projection": naive,
            "negative_clipped_gradient_projection": before["negative_clipped_gradient_projection"],
            "layer_negative_gradient_projections": before["layer_projections"],
            "adam_common_denominator_term_projections": terms,
            "adam_mathematical_delta_projection": before["adam_mathematical_delta_projection"],
            "actual_parameter_delta_projection": before["actual_delta_projection"],
            "parameter_storage_and_arithmetic_projection_residual": storage,
            "nonlinear_fixed_gap_residual": nonlinear,
            "decomposition_closure_residual": change - (sum(terms.values()) + storage + nonlinear),
            "naive_direction_sign": sign(naive), "finite_adam_gap_change_sign": sign(change),
            "naive_vs_finite_adam_sign_mismatch": naive * change < 0,
            "naive_vs_adam_parameter_projection_sign_mismatch": naive * before["actual_delta_projection"] < 0,
            "counterfactuals": branches,
        })
    global_geometry = case["component_gradients"]["global"]
    pair_checks = []
    for pair in global_geometry["pairs"]:
        first, second = pair["first"], pair["second"]
        pairs = [(row["negative_component_gradient_projections"][first],
                  row["negative_component_gradient_projections"][second]) for row in probes]
        pair_checks.append({
            "first": first, "second": second, "global_cosine": pair["cosine"],
            "globally_opposed": pair["cosine"] is not None and pair["cosine"] < 0,
            "local_opposite_direction_probe_count": sum(a * b < 0 for a, b in pairs),
            "local_both_gap_increasing_count": sum(a > 0 and b > 0 for a, b in pairs),
            "local_both_gap_decreasing_count": sum(a < 0 and b < 0 for a, b in pairs),
            "local_zero_projection_count": sum(a == 0 or b == 0 for a, b in pairs),
        })
    actual_branch = resolved["actual"]
    return {
        "arm": case["arm"], "incoming_td_update": case["incoming_td_update"],
        "pre_update_td": case["pre_update_td"], "mc_weight": case["mc_weight"],
        "losses_before": case["losses_before"], "global_gradient_geometry": global_geometry,
        "layer_gradient_geometry": case["component_gradients"]["layers"],
        "global_alignment_vs_local_gap_directions": pair_checks,
        "actual_gradient_norm": actual_branch["gradient_norm"],
        "actual_clip_coefficient": actual_branch["clip_coefficient"],
        "actual_parameter_delta_norm": actual_branch["parameter_delta_norm_float64"],
        "actual_optimizer_steps_before": actual_branch["optimizer_steps_before"],
        "actual_optimizer_steps_after": actual_branch["optimizer_steps_after"],
        "adam_decomposition": case["adam_decomposition"],
        "independent_optimizer_steps": case["independent_optimizer_steps"], "probes": probes,
    }


def aggregate_cases(cases):
    """Count only selected local state-update pairs, never successful episodes."""
    arms = {}
    flags = ("negative_to_positive_gap", "positive_to_negative_gap", "action_recovered_from_before",
             "action_regressed_from_before", "action_recovered_relative_to_actual", "action_regressed_relative_to_actual")
    for arm in sorted({case["arm"] for case in cases}):
        selected = [case for case in cases if case["arm"] == arm]
        probes = [probe for case in selected for probe in case["probes"]]
        branches = {}
        for branch in BRANCHES:
            values = [probe["counterfactuals"][branch] for probe in probes]
            branches[branch] = {flag + "_count": sum(row[flag] for row in values) for flag in flags}
            branches[branch].update(
                after_negative_gap_count=sum(row["after_greedy_gap"] < 0 for row in values),
                after_exact_tie_count=sum(row["after_greedy_gap"] == 0 for row in values),
                gap_higher_than_actual_count=sum(row["greedy_gap_difference_from_actual"] > 0 for row in values),
                gap_lower_than_actual_count=sum(row["greedy_gap_difference_from_actual"] < 0 for row in values),
                alias_case_count=sum(row["counterfactuals"][branch]["alias_of"] is not None
                                     for row in (case["probes"][0] for case in selected)))
        arms[arm] = {
            "selected_update_count": len(selected), "local_probe_update_pairs": len(probes),
            "independent_optimizer_steps": sum(case["independent_optimizer_steps"] for case in selected),
            "naive_vs_finite_adam_sign_mismatch_count": sum(row["naive_vs_finite_adam_sign_mismatch"] for row in probes),
            "naive_vs_adam_parameter_projection_sign_mismatch_count": sum(
                row["naive_vs_adam_parameter_projection_sign_mismatch"] for row in probes),
            "zero_naive_or_finite_change_count": sum(row["negative_actual_gradient_projection"] == 0
                                                     or row["actual_fixed_gap_change"] == 0 for row in probes),
            "old_first_moment_negative_gap_projection_count": sum(
                row["adam_common_denominator_term_projections"]["old_first_moment"] < 0 for row in probes),
            "counterfactuals": branches,
            "by_probe": {name: {
                "selected_update_count": sum(row["name"] == name for row in probes),
                "actual_after_negative_gap_count": sum(row["actual_after_greedy_gap"] < 0 for row in probes if row["name"] == name),
                "naive_vs_finite_adam_sign_mismatch_count": sum(row["naive_vs_finite_adam_sign_mismatch"] for row in probes if row["name"] == name),
            } for name in sorted({row["name"] for row in probes})},
        }
    return arms


def historical_contexts(cases, observations):
    """Link selected failures to existing outcomes without assigning causality."""
    failures, comparators = [], []
    for case in cases:
        historical = next((row for row in observations[case["arm"]]
                           if row["update"] == case["incoming_td_update"]), None)
        if historical is None:
            continue
        if case["incoming_td_update"] in (2001, 2010, 2200, 10000) and all(
                row["rollout"]["success"] for row in historical["scenes"]):
            comparators.append({"arm": case["arm"], "incoming_td_update": case["incoming_td_update"],
                                "historical_all_three_success": True,
                                "local_probe_gaps": {row["name"]: row["actual_after_greedy_gap"] for row in case["probes"]}})
        for scene in historical["scenes"]:
            rollout = scene["rollout"]
            if rollout["success"]:
                continue
            divergence = rollout["first_reference_action_divergence"]
            prefix = f"scene_{scene['seed']}_"
            name = prefix + ("actual_step_40" if scene["seed"] == 10002 else f"step_{divergence['step']}")
            probe = next(row for row in case["probes"] if row["name"] == name)
            context = {
                "arm": case["arm"], "incoming_td_update": case["incoming_td_update"], "scene_seed": scene["seed"],
                "historical_rollout": {key: rollout[key] for key in (
                    "success", "steps", "return", "damage", "budget_exhausted", "first_reference_action_divergence")},
                "diagnostic_probe": probe,
            }
            if scene["seed"] == 10002:
                context["earliest_reference_divergence_context"] = next(
                    row for row in case["probes"] if row["name"] == prefix + "step_37")
                context["selection_note"] = "Actual step40 separates successful and failed historical policies after an identical39-action prefix; step37 deviation also occurred in successful episodes. Neither observation alone proves a cause."
            failures.append(context)
    return failures, comparators


def analyze(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = read_json(run_dir / "summary.json")
    require(summary["source_hashes_unchanged"], "The run did not confirm unchanged historical inputs.")
    verified = {str(run_dir / "summary.json"): digest(run_dir / "summary.json")}
    for name, expected in summary["diagnostic_artifacts_sha256"].items():
        path = run_dir / name
        require(digest(path) == expected, f"Diagnostic artifact changed: {name}")
        verified[str(path)] = expected
    metadata_rows = read_json(run_dir / "probe_metadata.json")
    metadata = {row["name"]: row for row in metadata_rows}
    require(len(metadata) == len(metadata_rows) == 9, "Exactly nine distinct diagnostic states are required.")
    cases, keys = [], []
    for entry in summary["cases"]:
        path = run_dir / entry["file"]
        require(digest(path) == entry["sha256"], "A retained case file changed after diagnostics.")
        verified[str(path)] = entry["sha256"]
        case = read_json(path)
        key = (case["arm"], case["incoming_td_update"])
        require(key == (entry["arm"], entry["incoming_td_update"]), "Case metadata differs from its manifest.")
        keys.append(key)
        cases.append(summarize_case(case, metadata))
    expected = {(arm, update) for arm in summary["protocol"]["arms"]
                for update in summary["protocol"]["incoming_td_updates"]}
    require(len(keys) == len(set(keys)) == 28 and set(keys) == expected,
            "The complete two-arm, fourteen-update selection is required.")
    manifest_path = run_dir / "input_manifest.json"
    manifest = read_json(manifest_path)
    verified[str(manifest_path)] = digest(manifest_path)
    reference = Path(manifest["reference_run"])
    source_manifest_path = reference / "input_manifest.json"
    source_manifest = read_json(source_manifest_path)
    require(digest(source_manifest_path) == manifest["files_sha256"][str(source_manifest_path)],
            "The historical source manifest changed.")
    verified[str(source_manifest_path)] = digest(source_manifest_path)
    directories = {"mc_anchor": reference / "treatment/td",
                   "baseline": Path(source_manifest["reference_run"]) / "baseline/td"}
    observations = {}
    for arm, directory in directories.items():
        path = directory / "observations.json"
        require(digest(path) == manifest["files_sha256"][str(path)], "Historical physical outcome records changed.")
        verified[str(path)] = digest(path)
        observations[arm] = read_json(path)
    failures, comparators = historical_contexts(cases, observations)
    require(len(failures) == 8, "The historical selection must retain all eight failure contexts.")
    return {
        "run_dir": str(run_dir), "analysis_script_sha256": digest(Path(__file__)),
        "verified_input_sha256": verified, "case_count": len(cases), "probe_count": len(metadata),
        "probe_metadata": metadata_rows, "cases": cases, "selected_local_counts_by_arm": aggregate_cases(cases),
        "historical_failure_contexts": failures, "historical_successful_comparators": comparators,
        "run_accounting": summary["accounting"],
        "analysis_accounting": {"environment_steps": 0, "environment_resets": 0, "optimizer_updates": 0, "model_forwards": 0},
        "semantics": {
            "positive_local_projection": "A positive projection predicts increasing the fixed-before-competitor reference-action gap.",
            "negative_to_positive_recovery": "Strict before-gap<0 and after-max-other-gap>0; exact ties are separate. Action-recovery counts use actual argmax and can differ at ties.",
            "adam_terms": "Additive projections use the same actual post-update second-moment denominator and clipping coefficient. Independent ablations recompute their own moments and clipping.",
            "arm_comparison": "Matching absolute updates across arms generally have different prior parameters and moments; these cross-arm comparisons are descriptive, not same-state loss ablations.",
        },
        "limitations": ["Retrospectively selected updates and known states, not a representative sample or held-out evaluation.",
                        "Global gradient opposition does not determine a particular action gap.",
                        "One-step local ranking recovery does not establish episode success or durable training improvement.",
                        "Scene10002 actual step40 uses a successful-policy reference action, not a new expert demonstration.",
                        "All physical outcomes are reused historical evidence; this analysis performs no rollout."],
    }


def make_figure(analysis, path):
    """Create a monochrome diagnostic figure from retained numbers only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require(not path.exists(), f"Refusing to overwrite an existing diagnostic figure: {path}")
    contexts = analysis["historical_failure_contexts"]
    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1, 1, 1.1))
    labels = ["Old m", "TD", "Margin", "MC", "Round", "Observed"]
    hatches = ["//", "\\\\", "xx", "..", "++", None]
    row_labels, table_values = [], []
    table_branches = ("actual", "no_td", "no_margin", "no_mc", "zero_first_moment", "reset_adam")
    for index, context in enumerate(contexts):
        probe = context["diagnostic_probe"]
        values = [probe["adam_common_denominator_term_projections"][name] for name in TERMS]
        values.append(probe["actual_fixed_gap_change"])
        axis = figure.add_subplot(grid[index // 4, index % 4])
        bars = axis.bar(range(len(values)), values, color="white", edgecolor="black", linewidth=0.8)
        for bar, hatch in zip(bars, hatches, strict=True):
            if hatch is None:
                bar.set_facecolor("black")
            else:
                bar.set_hatch(hatch)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        title = f"{context['arm']} TD{context['incoming_td_update']}\n{probe['name'].replace('scene_', 'scene ')}"
        axis.set_title(title, fontsize=9)
        axis.set_ylabel("Fixed-gap contribution / change", fontsize=8)
        axis.tick_params(axis="y", labelsize=8)
        row_labels.append(f"{context['arm']} {context['incoming_td_update']} / {context['scene_seed']}")
        table_values.append([f"{probe['counterfactuals'][name]['after_greedy_gap']:.5f}" for name in table_branches])
    axis = figure.add_subplot(grid[2, :])
    axis.axis("off")
    table = axis.table(cellText=table_values, rowLabels=row_labels,
                       colLabels=[name.replace("_", " ") for name in table_branches], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    table.add_cell(0, -1, width=table[1, -1].get_width(), height=table[0, 0].get_height(), text="")
    for (row, _), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.visible_edges = "TB" if row == 0 else "B" if row == len(contexts) else ""
        cell.set_linewidth(0.8)
        cell.get_text().set_color("black")
    axis.set_title("After-step max-other gaps for independent one-step branches (local state only)", fontsize=10)
    figure.suptitle("Retrospective failure contexts: common-denominator Adam projections and measured local changes", fontsize=12)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.figure is not None and args.figure.exists()):
        raise FileExistsError("Refusing to overwrite existing diagnostic analysis artifacts.")
    analysis = analyze(args.run_dir)
    serialized = json.dumps(analysis, indent=2, allow_nan=False) + "\n"
    require(len(serialized.encode("utf-8")) <= MAX_BYTES, "Analysis exceeds the 8 MiB artifact cap.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    if args.figure is not None:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        make_figure(analysis, args.figure)
    print(f"Analyzed {analysis['case_count']} historical cases and {analysis['probe_count']} local states.")


if __name__ == "__main__":
    main()
