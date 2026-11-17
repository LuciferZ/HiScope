"""Generate independent executable configs; never launch expensive runs implicitly.

Run from code/: python scripts/make_experiments.py --help
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrmem.data import read_json, write_json, locomo_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/paper.json")
    p.add_argument("--matrix", default="configs/experiments.json")
    p.add_argument("--output", default="experiments")
    p.add_argument("--folds", choices=["fixed", "tenfold"], default="fixed")
    p.add_argument("--granularity", choices=["coarse", "fine"], default="coarse")
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--grid", action="store_true", help="Expand validation grids; can create thousands of runs")
    p.add_argument("--sft-root", default="checkpoints", help="Contains fold_N/seed_N/{scope_sft,full_sft}/best")
    args = p.parse_args()
    base, matrix = read_json(args.base), read_json(args.matrix)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for fold in (range(10) if args.folds == "tenfold" else [None]):
        fold_name = "fixed" if fold is None else f"fold_{fold}"
        split_path = output / "splits" / (fold_name+".json")
        write_json(split_path, locomo_split(fold))
        for seed, condition in itertools.product(matrix["seeds"], matrix["conditions"]):
            if args.conditions and condition["name"] not in args.conditions:
                continue
            grid = [{}]
            if args.grid:
                keys = ["k_topic"] if condition.get("method") == "rrf" else (
                    ["beta_kl", "length_weight", "learning_rate", "updates"] if condition["stage"] == "grpo" else [])
                grid = [dict(zip(keys, values)) for values in itertools.product(*(matrix["validation_grid"][k] for k in keys))]
            for variant, hp in enumerate(grid):
                config = copy.deepcopy(base)
                config.update(split=str(split_path), seed=seed, condition=condition["name"],
                              method=condition.get("method", "learned"), zero_shot=condition.get("zero_shot", False))
                config["training"]["seed"] = seed
                if condition["protocol"] == "adaptive":
                    config["models"]["granularity_gate"] = dict(
                        path=str(Path(args.sft_root)/fold_name/f"seed_{seed}"/"granularity_gate"/"best"),
                        device=base["models"]["answer"].get("device", "cpu"), dtype="bfloat16")
                config["interface"].update(view=condition["view"], protocol=condition["protocol"],
                                            granularity=args.granularity)
                if "length_weight" in condition:
                    config["training"]["length_weight"] = condition["length_weight"]
                if condition.get("initialization") != "base" and condition.get("method") != "rrf":
                    kind = "scope_sft" if condition["protocol"] == "scope" else "full_sft"
                    config["models"]["policy"]["path"] = str(Path(args.sft_root)/fold_name/f"seed_{seed}"/kind/"best")
                for key, value in hp.items():
                    if key == "k_topic":
                        config[key] = value
                    else:
                        config["training"][key] = value
                run_name = f"{fold_name}/{args.granularity}/{condition['name']}/seed_{seed}/v{variant}"
                config["output"] = str(output/"runs"/run_name)
                path = output/"configs"/(run_name+".json")
                write_json(path, config)
                task = dict(condition=condition["name"], fold=fold, seed=seed, hyperparameters=hp,
                            command=[sys.executable, "-m", "mrmem", condition["stage"], "--config", str(path)])
                if condition["stage"] == "grpo":
                    evaluation = copy.deepcopy(config)
                    evaluation["models"]["policy"]["path"] = config["output"]+"/best"
                    evaluation["output"] = config["output"]+"_test"
                    eval_path = output/"configs"/(run_name+"_test.json")
                    write_json(eval_path, evaluation)
                    task["evaluation_command"] = [sys.executable, "-m", "mrmem", "evaluate", "--config", str(eval_path)]
                tasks.append(task)
    write_json(output/"tasks.json", tasks)
    print(f"Wrote {len(tasks)} tasks to {output}; select hyperparameters on dev before running test commands.")


if __name__ == "__main__":
    main()
