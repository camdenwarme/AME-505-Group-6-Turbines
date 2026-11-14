"""Select a team-advantage weight from matched, completed pilot runs.

Selection uses mean final greedy farm energy over matched seeds. The output is
a reproducible pilot choice, not evidence of a globally optimal weight.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean


def _finite(value, field, directory):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{directory}: invalid {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{directory}: nonfinite {field}")
    return number


def read_pilot(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with (directory / "training_metrics.csv").open(newline="", encoding="utf-8") as stream:
        training = list(csv.DictReader(stream))
    with (directory / "evaluation_metrics.csv").open(newline="", encoding="utf-8") as stream:
        evaluations = list(csv.DictReader(stream))
    if not training:
        raise ValueError(f"{directory}: no completed training episodes")
    episode = int(training[-1]["episode"])
    expected = int(config["num_env_steps"]) // (int(config["episode_length"]) *
                                                  int(config["n_rollout_threads"]))
    if episode != expected or len(training) != expected:
        raise ValueError(f"{directory}: incomplete training ({episode}/{expected} episodes)")
    greedy = [row for row in evaluations if row.get("policy") == "greedy"]
    before = [row for row in greedy if int(row["episode"]) == 0]
    final = [row for row in greedy if int(row["episode"]) == episode]
    if len(before) != 1 or len(final) != 1:
        raise ValueError(f"{directory}: require exactly one initial and final greedy evaluation")
    before_scenario, final_scenario = before[0].get("scenario_id", ""), final[0].get("scenario_id", "")
    if before_scenario != final_scenario:
        raise ValueError(f"{directory}: initial and final evaluations used different wind scenarios")
    weight = _finite(config.get("team_reward_weight"), "team_reward_weight", directory)
    if not 0 <= weight <= 1:
        raise ValueError(f"{directory}: team weight outside [0, 1]")
    def metrics(row):
        fields = ["farm_energy_mwh"] + sorted(
            (key for key in row if re.fullmatch(r"turbine\d+_energy_mwh", key)),
            key=lambda key: int(key.split("_")[0][7:]))
        expected_count = int(config.get("num_agents", len(fields) - 1))
        if len(fields) - 1 != expected_count:
            raise ValueError(f"{directory}: missing per-turbine energy fields")
        return {field: _finite(row.get(field), field, directory) for field in fields}
    return {"path": str(directory), "config": config, "weight": weight,
            "seed": int(config["seed"]), "training_episodes": episode,
            "evaluation_scenario_id": final_scenario,
            "environment_steps": int(training[-1]["environment_steps"]),
            "greedy_evaluation_count": len(greedy), "initial_greedy": metrics(before[0]),
            "final_greedy": metrics(final[0])}


def _config_signature(config, *, across_seeds):
    """Keep all treatment settings, optionally excluding independent RNG seeds."""
    omitted = {"team_reward_weight", "experiment_name"}
    if across_seeds:
        omitted.update(("seed", "wind_seed", "eval_wind_seed"))
    signature = {key: value for key, value in config.items() if key not in omitted}
    if across_seeds:
        for key in ("training_wind_config", "evaluation_wind_config"):
            if isinstance(signature.get(key), dict):
                signature[key] = {name: value for name, value in signature[key].items() if name != "seed"}
    return signature


def _require_matching_configs(pilots, *, across_seeds):
    reference = _config_signature(pilots[0]["config"], across_seeds=across_seeds)
    for pilot in pilots[1:]:
        common = _config_signature(pilot["config"], across_seeds=across_seeds)
        differences = sorted(key for key in set(reference) | set(common)
                             if reference.get(key) != common.get(key))
        if differences:
            raise ValueError(f"{pilot['path']}: unmatched configuration fields: {', '.join(differences)}")


def select(pilots):
    if not pilots:
        raise ValueError("Pass at least one pilot run directory")
    # Across independent seeds permit model/wind/evaluation RNG seeds to vary,
    # including the repeated seeds in serialized wind configs. Every other
    # rollout, optimizer and wind-distribution setting must remain identical.
    _require_matching_configs(pilots, across_seeds=True)
    groups = {}
    for pilot in pilots:
        group = groups.setdefault(pilot["weight"], {})
        if pilot["seed"] in group:
            raise ValueError(f"Duplicate seed {pilot['seed']} for weight {pilot['weight']}")
        group[pilot["seed"]] = pilot
    expected_seeds = set(next(iter(groups.values())))
    if any(set(group) != expected_seeds for group in groups.values()):
        raise ValueError("All candidate weights must use the same set of seeds")
    for seed in expected_seeds:
        # Candidate weights for one seed must share *exactly* the same RNG
        # settings, not merely the same realized evaluation scenario string.
        _require_matching_configs([group[seed] for group in groups.values()], across_seeds=False)
        scenarios = {group[seed].get("evaluation_scenario_id", "") for group in groups.values()}
        if len(scenarios) != 1:
            raise ValueError(f"Candidate weights used different evaluation wind scenarios for seed {seed}")
    summaries = []
    for weight, group in sorted(groups.items()):
        runs = [group[seed] for seed in sorted(group)]
        summaries.append({"weight": weight, "seed_count": len(runs),
                          "seeds": [run["seed"] for run in runs],
                          "mean_initial_farm_energy_mwh": mean(r["initial_greedy"]["farm_energy_mwh"] for r in runs),
                          "mean_final_farm_energy_mwh": mean(r["final_greedy"]["farm_energy_mwh"] for r in runs),
                          "runs": [{k: v for k, v in run.items() if k != "config"} for run in runs]})
    best_energy = max(group["mean_final_farm_energy_mwh"] for group in summaries)
    winning_weights = [group["weight"] for group in summaries
                       if group["mean_final_farm_energy_mwh"] == best_energy]
    selected_weight = min(winning_weights)
    return {"selected_weight": selected_weight, "tied_winning_weights": winning_weights,
            "criterion": "highest mean final greedy farm energy across matched seeds",
            "tie_break": "lowest weight", "candidate_count": len(summaries),
            "seeds_per_candidate": len(expected_seeds),
            "training_episodes_per_pilot": pilots[0]["training_episodes"],
            "limitations": ["Selection is based on short pilots and does not establish a global optimum.",
                            "It uses final greedy farm energy; individual turbine outcomes can differ.",
                            "The same pilot evaluations select and assess the weight; validate on new seeds and conditions."],
            "candidates": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Path for machine-readable selection JSON")
    args = parser.parse_args()
    try:
        result = select([read_pilot(path) for path in args.run_dirs])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("weight  seeds  episodes  initial MWh  final MWh")
    for item in result["candidates"]:
        print(f"{item['weight']:>6g}  {item['seed_count']:>5}  "
              f"{result['training_episodes_per_pilot']:>8}  "
              f"{item['mean_initial_farm_energy_mwh']:>11.6f}  "
              f"{item['mean_final_farm_energy_mwh']:>9.6f}")
    print(f"Selected pilot weight: {result['selected_weight']:g}")
    if len(result["tied_winning_weights"]) > 1:
        print(f"Exact tie among {result['tied_winning_weights']}; selected lowest weight.")
    print(f"Selection record: {args.output}")
    print("Short pilots are a screening result; validate with new seeds and wind conditions.")


if __name__ == "__main__":
    main()
