"""Per-question evaluation, paired cluster bootstrap and factorial decomposition."""
from collections import defaultdict
from dataclasses import asdict, replace
import math
import random
import time

from .environment import Environment, conventional_action
from .rewards import correctness
from .rollout import rollout


def evaluate(questions, repositories, policy, answer, ranker, interface, judge=None,
             method="learned", k_topic=3, seed=2027, condition="MRMem"):
    rows = []
    for q in questions:
        if not q.evaluable:
            continue
        start = time.perf_counter()
        if method == "learned":
            t = rollout(q, repositories[q.owner_id], policy, answer, ranker, interface, judge, False)
            row = asdict(t)
            row.pop("actions")
        elif method == "rrf":
            # Deterministic rankers have no policy action budget or policy token cost.
            env = Environment(repositories[q.owner_id], q.query, ranker, answer.count,
                              replace(interface, action_budget=10**9))
            action = conventional_action(env, k_topic)
            transition = env.step(action)
            if not transition["valid"]:
                raise ValueError(transition["error"])
            prediction = answer(q.query, env.memory)
            row = dict(prediction=prediction, correct=correctness(q, prediction, judge, False),
                       memory_tokens=answer.count(env.memory), calls=1, invalid=0,
                       policy_input_tokens=0, policy_output_tokens=0,
                       answer_input_tokens=answer.last_input_tokens,
                       answer_output_tokens=answer.last_output_tokens,
                       evidence_ids=list(env.evidence))
        else:
            raise ValueError("method must be learned or rrf (zero-shot uses learned with a base checkpoint)")
        row.update(question_id=q.id, owner_id=q.owner_id, dataset=q.dataset, category=q.category,
                   condition=condition, seed=seed, latency=time.perf_counter()-start)
        row["online_tokens"] = sum(row[k] for k in ("policy_input_tokens", "policy_output_tokens",
                                                     "answer_input_tokens", "answer_output_tokens"))
        rows.append(row)
    if not rows:
        raise ValueError("No evaluable questions")
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("Empty results")
    return {"accuracy": sum(r["correct"] for r in rows)/len(rows),
            "memory_tokens": sum(r["memory_tokens"] for r in rows)/len(rows),
            "online_tokens": sum(r["online_tokens"] for r in rows)/len(rows),
            "questions": len(rows)}


def paired_contrast(rows, weights, metric="correct", resamples=10000, seed=2027):
    """Average paired seeds per question, then bootstrap owners with QA weights.

    `weights` supports gaps, factorial interactions and SFT/RL decomposition.
    Deterministic cells should be repeated with the same three seed labels.
    """
    import numpy as np
    cells = {name: {} for name in weights}
    for r in rows:
        if r["condition"] in cells:
            key = (r["owner_id"], r["question_id"], r["seed"])
            if key in cells[r["condition"]]:
                raise ValueError("Duplicate question/seed/condition; analyze datasets and folds separately")
            cells[r["condition"]][key] = float(r[metric])
    keys = set(next(iter(cells.values())))
    if not keys or any(set(cell) != keys for cell in cells.values()):
        raise ValueError("Paired contrast requires identical owner/question/seed coverage")
    values = defaultdict(list)
    seed_sets = defaultdict(set)
    for owner, question, s in sorted(keys):
        values[owner, question].append(sum(weights[name]*cell[owner, question, s] for name, cell in cells.items()))
        seed_sets[owner, question].add(s)
    if len({tuple(sorted(s)) for s in seed_sets.values()}) != 1:
        raise ValueError("All questions must have identical seed coverage")
    owners = defaultdict(list)
    for (owner, _), scores in values.items():
        owners[owner].append(sum(scores)/len(scores))
    names = sorted(owners)
    totals = np.array([sum(owners[o]) for o in names])
    counts = np.array([len(owners[o]) for o in names])
    point = float(totals.sum()/counts.sum())
    result = dict(estimate=point, clusters=len(names), questions=int(counts.sum()),
                  seeds=sorted(next(iter(seed_sets.values()))), per_owner={o: sum(owners[o])/len(owners[o]) for o in names})
    if len(names) <= 2:
        result["ci95"] = None  # paper's two-history split is descriptive
    else:
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, len(names), size=(resamples, len(names)))
        estimates = totals[samples].sum(1)/counts[samples].sum(1)
        result["ci95"] = np.quantile(estimates, [.025, .975]).tolist()
    npos = int((totals > 0).sum())
    nneg = int((totals < 0).sum())
    n = npos+nneg
    result["sign_test"] = dict(positive=npos, negative=nneg, ties=len(names)-n,
        nominal_p=min(1.0, 2*sum(math.comb(n, k) for k in range(min(npos, nneg)+1))/(2**n)) if n else 1.0)
    return result


def factorial_analysis(rows):
    result = {"interaction": paired_contrast(rows, {"HL": 1, "HC": -1, "FL": -1, "FC": 1})}
    if {"HL_SFT", "FL_SFT"} <= {r["condition"] for r in rows}:
        result["sft_interaction"] = paired_contrast(rows, {"HL_SFT": 1, "HC": -1, "FL_SFT": -1, "FC": 1})
        result["rl_increment"] = paired_contrast(rows, {"HL": 1, "FL": -1, "HL_SFT": -1, "FL_SFT": 1})
        total = result["interaction"]["estimate"]
        result["rl_share_point_estimate"] = result["rl_increment"]["estimate"]/total if total else None
    return result


def choose_validation_candidate(candidates, tolerance=.01):
    """Each candidate contains validation accuracy and mean memory_tokens."""
    if not candidates:
        raise ValueError("No validation candidates")
    best = max(c["accuracy"] for c in candidates)
    eligible = [c for c in candidates if c["accuracy"] >= best-tolerance]
    return min(eligible, key=lambda c: (c["memory_tokens"], -c["accuracy"]))
