"""Run with python -m mrmem from code/. Paths in configs are relative to code/."""
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json

from .data import (Question, Repository, read_json, read_jsonl, write_json, write_jsonl,
                   locomo_split, personamem_split, validate_split)
from .environment import Interface, Environment
from .evaluation import evaluate, summarize, factorial_analysis, paired_contrast, choose_validation_candidate
from .models import HFPolicy, HFAnswer, HFEmbedding, HTTPJudge
from .ranking import Ranker
from .training import TrainConfig, train_grpo, train_sft, seed_all


def load_dataset(config, partition):
    split = read_json(config["split"])
    validate_split(split)
    all_questions = [Question(**r) for r in read_jsonl(config["questions"])]
    identities = [(q.owner_id, q.id) for q in all_questions]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate question IDs within owner")
    questions = [q for q in all_questions if q.owner_id in split[partition]]
    if not questions:
        raise ValueError(f"No questions in {partition}")
    repositories = {owner: Repository.load(Path(config["repositories"])/f"{owner}.json")
                    for owner in {q.owner_id for q in questions}}
    for owner, repo in repositories.items():
        if owner != repo.owner_id:
            raise ValueError("Repository filename/owner mismatch")
    for q in questions:
        if not set(q.support_ids) <= repositories[q.owner_id].leaves.keys():
            # Evidence-sensitivity runs can lose annotated support; require explicit opt-in.
            if not config.get("allow_missing_support", False):
                raise ValueError("Question has missing support IDs; verify mapping or enable sensitivity mode")
    return questions, repositories


def load_components(config, with_policy=True):
    models = config["models"]
    policy = HFPolicy(**models["policy"]) if with_policy else None
    answer = HFAnswer(**models["answer"])
    if config.get("interface", {}).get("protocol") == "adaptive":
        answer.granularity_gate = HFPolicy(**models["granularity_gate"]).freeze()
    embed = HFEmbedding(**models["embedding"])
    judge = HTTPJudge(**config["judge"]) if config.get("judge") else None
    return policy, answer, Ranker(embed), judge


def save_provenance(output, config):
    fingerprints = {}
    for key in ("questions", "split", "demonstrations"):
        if config.get(key):
            fingerprints[key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    paper_root = Path(__file__).resolve().parents[2]
    for name in ("AnonymousSubmission2027_final.tex", "SupplementaryAppendix_final.tex"):
        path = paper_root / name
        if path.exists():
            fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(Path(output)/"provenance.json", dict(config=config, sha256=fingerprints))


def run_training(config, stage):
    seed = config.get("training", {}).get("seed", 2027)
    seed_all(seed)
    interface = Interface(**config.get("interface", {}))
    output = config["output"]
    if Path(output).exists():
        raise FileExistsError("Choose a fresh output directory; existing training runs are never overwritten")
    if stage == "sft":
        split = read_json(config["split"])
        validate_split(split)
        rows = read_jsonl(config["demonstrations"])
        train = [r for r in rows if r["owner_id"] in split["train"]]
        dev = [r for r in rows if r["owner_id"] in split["dev"]]
        # Held-out test owners cannot be used as either train or validation examples.
        policy = HFPolicy(**config["models"]["policy"])
        train_sft(policy, train, dev, output, seed=seed,
                  epochs=2 if interface.protocol == "scope" else 3, **config.get("sft", {}))
    else:
        train, repos = load_dataset(config, "train")
        dev, dev_repos = load_dataset(config, "dev")
        policy, answer, ranker, judge = load_components(config)
        # Initialization and reference always share a checkpoint (SFT or base).
        ref_config = dict(config["models"]["policy"])
        ref_config.update(device=config.get("reference_device", ref_config.get("device", "cpu")),
                          checkpointing=False)
        reference = HFPolicy(**ref_config).freeze()
        def validate(current):
            return summarize(evaluate(dev, dev_repos, current, answer, ranker, interface, judge, seed=seed))
        train_grpo(policy, reference, train, repos, answer, ranker, interface, output,
                   TrainConfig(**config.get("training", {})), judge, validate)
    save_provenance(output, config)


def run_evaluation(config, partition="test"):
    seed = config.get("seed", config.get("training", {}).get("seed", 2027))
    seed_all(seed)
    method = config.get("method", "learned")
    questions, repos = load_dataset(config, partition)
    policy, answer, ranker, judge = load_components(config, method != "rrf")
    if config.get("zero_shot", False):
        # Paper comparator uses temperature zero; learned and SFT-only evaluation use 0.7.
        original = policy.sample
        policy.sample = lambda prompt, max_tokens=256: original(prompt, max_tokens, greedy=True)
    rows = evaluate(questions, repos, policy, answer, ranker, Interface(**config.get("interface", {})),
                    judge, method, config.get("k_topic", 3), seed, config.get("condition", "MRMem"))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output/"predictions.jsonl", rows)
    write_json(output/"metrics.json", summarize(rows))
    save_provenance(output, config)
    print(json.dumps(summarize(rows), indent=2))


def compile_demonstrations(config, source, destination):
    """Convert human-corrected action traces into answer-free policy training states."""
    # IDs need not be globally unique: use explicit owner in teacher traces.
    questions = {(q["owner_id"], q["id"]): Question(**q) for q in read_jsonl(config["questions"])}
    policy, answer, ranker, _ = load_components(config)
    rows = []
    for trace in read_jsonl(source):
        q = questions[trace["owner_id"], trace["question_id"]]
        repo = Repository.load(Path(config["repositories"])/f"{q.owner_id}.json")
        interface = Interface(**trace.get("interface", config.get("interface", {})))
        env = Environment(repo, q.query, ranker, answer.count, interface)
        env.action_counter = policy.count
        for action in trace["actions"]:
            if env.done:
                raise ValueError("Demonstration continues after stop")
            text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            row = dict(owner_id=q.owner_id, question_id=q.id, prompt=env.prompt(), completion=text,
                       interface=asdict(interface))
            transition = env.step(text)
            if not transition["valid"]:
                raise ValueError(f"Invalid demonstration {q.id}: {transition['error']}")
            rows.append(row)
        if not env.done:
            raise ValueError("Incomplete teacher trajectory")
    write_jsonl(destination, rows)


def main():
    parser = argparse.ArgumentParser(description="MRMem training and experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("sft", "grpo", "evaluate", "compile-demos", "construct", "resize"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "evaluate":
            p.add_argument("--partition", choices=["dev", "test"], default="test")
        if command == "compile-demos":
            p.add_argument("--traces", required=True)
            p.add_argument("--output", required=True)
    p = sub.add_parser("splits")
    p.add_argument("--dataset", choices=["locomo", "personamem"], default="locomo")
    p.add_argument("--fold", type=int)
    p.add_argument("--owners", help="JSON list of PersonaMem user IDs")
    p.add_argument("--output", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--weights", help='JSON contrast, e.g. {"MRMem":1,"fixed_fine":-1}')
    p.add_argument("--metric", default="correct")
    p.add_argument("--output", required=True)
    p = sub.add_parser("select")
    p.add_argument("--candidates", required=True, help="JSON list of validation metrics/configs")
    p.add_argument("--tolerance", type=float, default=.01)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in {"sft", "grpo"}:
        run_training(read_json(args.config), args.command)
    elif args.command == "evaluate":
        run_evaluation(read_json(args.config), args.partition)
    elif args.command == "compile-demos":
        compile_demonstrations(read_json(args.config), args.traces, args.output)
    elif args.command == "splits":
        split = locomo_split(args.fold) if args.dataset == "locomo" else personamem_split(read_json(args.owners))
        write_json(args.output, split)
    elif args.command == "analyze":
        rows = [r for path in args.predictions for r in read_jsonl(path)]
        result = paired_contrast(rows, json.loads(args.weights), args.metric) if args.weights else factorial_analysis(rows)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "select":
        write_json(args.output, choose_validation_candidate(read_json(args.candidates), args.tolerance))
    elif args.command == "construct":
        from .data import Source
        from .preparation import construct_repository
        config = read_json(args.config)
        model = HFPolicy(**config["extractor"]).freeze()
        embedding = HFEmbedding(**config["embedding"])
        source = read_json(config["sources"])
        generator = lambda prompt: model.sample(prompt, max_tokens=config.get("max_tokens", 4096), greedy=True).text
        repo = construct_repository(source["owner_id"], [Source(**s) for s in source["sources"]], generator, embedding)
        repo.save(config["output"])
    elif args.command == "resize":
        from .preparation import resize_repository
        config = read_json(args.config)
        repo, manifest = resize_repository(Repository.load(config["repository"]),
            size=config.get("size", 500), seed=config.get("seed", 2027),
            protected_ids=config.get("protected_ids"),
            donors=[Repository.load(p) for p in config.get("donors", [])],
            allowed_donor_owners=config.get("allowed_donor_owners", []),
            random_subsample=config.get("random_subsample", False))
        repo.save(config["output"])
        write_json(config["output"]+".manifest.json", manifest)
