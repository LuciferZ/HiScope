"""Full-parameter SFT and trajectory-level GRPO, with sequential microbatch accumulation."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import random

from .data import write_json
from .models import ActionSample
from .rollout import rollout


@dataclass
class TrainConfig:
    seed: int = 2027
    group_size: int = 8
    groups_per_batch: int = 8
    updates: int = 1000
    update_epochs: int = 4
    minibatch_size: int = 8
    learning_rate: float = 1e-6
    warmup_updates: int = 100
    clip_epsilon: float = 0.2
    beta_kl: float = 0.01
    length_weight: float = 0.7
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    validate_every: int = 50


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_advantages(rewards):
    import torch
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Expected [question groups, trajectories], G >= 2")
    return (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(
        -1, correction=0, keepdim=True).clamp_min(1e-6)


def grpo_loss(log_prob, old_log_prob, advantage, kl, epsilon=0.2, beta=0.01):
    """Whole-trajectory ratio, not mean token ratio; KL is a token SUM."""
    import torch
    log_ratio = log_prob.float() - old_log_prob
    if not torch.isfinite(log_ratio).all() or (log_ratio > 80).any():
        raise FloatingPointError("Trajectory ratio overflow; reduce learning rate/check old policy")
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)
    loss = -surrogate + beta * kl
    return loss, {"clip_fraction": float(((ratio < 1-epsilon) | (ratio > 1+epsilon)).float().mean()),
                  "abs_log_ratio": float(log_ratio.detach().abs().mean()),
                  "kl": float(kl.detach().mean())}


def _append(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_grpo(policy, reference, questions, repositories, answer, ranker, interface,
               output, config=None, judge=None, validate=None):
    import torch
    config = config or TrainConfig()
    if not questions or config.group_size < 2 or config.groups_per_batch < 1:
        raise ValueError("Empty training set or invalid GRPO group settings")
    if policy.temperature != reference.temperature:
        raise ValueError("Policy/reference temperatures must match")
    if policy.tokenizer.get_vocab() != reference.tokenizer.get_vocab():
        raise ValueError("Policy/reference vocabularies differ")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "train_config.json", asdict(config))
    seed_all(config.seed)
    rng = random.Random(config.seed)
    reference.freeze()
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best, best_tokens = -math.inf, math.inf
    for update in range(1, config.updates + 1):
        groups = []
        for q in rng.choices(questions, k=config.groups_per_batch):
            groups.append([rollout(q, repositories[q.owner_id], policy, answer, ranker, interface,
                           judge, True, config.length_weight) for _ in range(config.group_size)])
        advantages = group_advantages([[t.reward for t in g] for g in groups]).flatten().tolist()
        trajectories = [t for g in groups for t in g]
        lr = config.learning_rate * min(1.0, update / max(1, config.warmup_updates))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        diagnostics = []
        for _ in range(config.update_epochs):
            order = list(range(len(trajectories)))
            rng.shuffle(order)
            for start in range(0, len(order), config.minibatch_size):
                batch = order[start:start+config.minibatch_size]
                optimizer.zero_grad(set_to_none=True)
                for index in batch:
                    trajectory = trajectories[index]
                    scored = [policy.score(a, reference) for a in trajectory.actions]
                    logp = sum(x[0] for x in scored)
                    kl = sum(x[1] for x in scored)
                    old = sum(a.old_log_prob for a in trajectory.actions)
                    loss, diagnostic = grpo_loss(logp, old, advantages[index], kl,
                                                 config.clip_epsilon, config.beta_kl)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite GRPO loss")
                    (loss / len(batch)).backward()
                    diagnostics.append(diagnostic)
                torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config.max_grad_norm,
                                               error_if_nonfinite=True)
                optimizer.step()
        log = dict(update=update, lr=lr,
                   reward=sum(t.reward for t in trajectories)/len(trajectories),
                   accuracy=sum(t.correct for t in trajectories)/len(trajectories),
                   memory_tokens=sum(t.memory_tokens for t in trajectories)/len(trajectories),
                   invalid_rate=sum(t.invalid for t in trajectories)/sum(t.calls for t in trajectories),
                   all_incorrect_groups=sum(not any(t.correct for t in g) for g in groups)/len(groups))
        log.update({key: sum(d[key] for d in diagnostics)/len(diagnostics) for key in diagnostics[0]})
        if validate is not None and (update % config.validate_every == 0 or update == config.updates):
            # Validation sampling must not perturb subsequent training trajectories.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(config.seed)
                metrics = validate(policy)
            log["validation"] = metrics
            # Checkpoint selection by validation accuracy; tie-break by memory tokens.
            if (metrics["accuracy"], -metrics["memory_tokens"]) > (best, -best_tokens):
                best, best_tokens = metrics["accuracy"], metrics["memory_tokens"]
                policy.save(output / "best")
        _append(output / "train.jsonl", log)
        print(json.dumps(log), flush=True)
    policy.save(output / "last")
    torch.save(dict(optimizer=optimizer.state_dict(), update=config.updates,
                    torch_rng=torch.get_rng_state(), python_rng=rng.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
               output / "optimizer.pt")


def encode_demonstration(policy, row):
    prompt_ids = policy.encode_prompt(row["prompt"])
    tokens = policy.tokenizer.encode(row["completion"], add_special_tokens=False)
    if policy.tokenizer.eos_token_id is not None:
        tokens.append(policy.tokenizer.eos_token_id)
    if len(tokens) > 256 or len(prompt_ids) + len(tokens) > policy.limit:
        raise ValueError("Demonstration exceeds action/context budget")
    return ActionSample(row["prompt"], row["completion"], prompt_ids, tokens)


def train_sft(policy, train_rows, dev_rows, output, seed=2027, epochs=3, batch_size=16,
              learning_rate=1e-5):
    import torch
    if not train_rows or not dev_rows:
        raise ValueError("SFT requires train and held-out owner-level validation demonstrations")
    if {r["owner_id"] for r in train_rows} & {r["owner_id"] for r in dev_rows}:
        raise ValueError("SFT owner leakage")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    rng = random.Random(seed)
    train = [encode_demonstration(policy, r) for r in train_rows]
    dev = [encode_demonstration(policy, r) for r in dev_rows]
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=learning_rate, weight_decay=0)
    best = math.inf
    for epoch in range(1, epochs+1):
        rng.shuffle(train)
        for start in range(0, len(train), batch_size):
            batch = train[start:start+batch_size]
            denominator = sum(len(s.token_ids) for s in batch)
            optimizer.zero_grad(set_to_none=True)
            for sample in batch:
                loss, _ = policy.sft_loss(sample)
                (loss / denominator).backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            total, count = 0.0, 0
            for sample in dev:
                loss, n = policy.sft_loss(sample)
                total += loss.item()
                count += n
        metric = total/count
        _append(output / "sft.jsonl", dict(epoch=epoch, validation_loss=metric))
        if metric < best:
            best = metric
            policy.save(output / "best")
        else:
            break  # patience one, specified implementation choice
    policy.save(output / "last")
