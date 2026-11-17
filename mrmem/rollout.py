from .models import Trajectory
from .environment import Environment
from .rewards import correctness, reward
from dataclasses import replace
import json


def rollout(question, repository, policy, answer, ranker, interface, judge=None,
            training=True, length_weight=0.7):
    if question.owner_id != repository.owner_id:
        raise ValueError("Question/repository owner mismatch")
    result = Trajectory()
    if interface.protocol == "adaptive":
        gate = getattr(answer, "granularity_gate", None)
        if gate is None:
            raise ValueError("Adaptive retrieval requires a frozen, separately SFT-trained granularity gate")
        metadata = Environment(repository, question.query, ranker, answer.count,
                               replace(interface, protocol="fixed_coarse")).metadata()
        prompt = ("Decide whether answering requires raw source messages. Return only coarse or fine.\n"
                  f"Question: {question.query}\nParents: {json.dumps(metadata, ensure_ascii=False)}\nDecision:\n")
        decision = gate.sample(prompt, max_tokens=8, greedy=True)
        if decision.text not in {"coarse", "fine"}:
            raise ValueError("Granularity gate must return exactly coarse or fine")
        interface = replace(interface, protocol="fixed_"+decision.text, granularity=decision.text)
        result.gate_calls = 1
        result.policy_input_tokens += len(decision.prompt_ids)
        result.policy_output_tokens += len(decision.token_ids)
    env = Environment(repository, question.query, ranker, answer.count, interface)
    env.action_counter = policy.count
    while not env.done:
        sample = policy.sample(env.prompt(), interface.action_budget)
        # Old scores are taken before any optimization, from exactly the sampling distribution.
        if training:
            with policy.torch.no_grad():
                sample.old_log_prob = float(policy.score(sample)[0].item())
        result.actions.append(sample)
        result.policy_input_tokens += len(sample.prompt_ids)
        result.policy_output_tokens += len(sample.token_ids)
        transition = env.step(sample.text)
        result.invalid += int(not transition["valid"])
    result.prediction = answer(question.query, env.memory)
    result.correct = correctness(question, result.prediction, judge, training)
    result.memory_tokens = answer.count(env.memory)
    result.reward = reward(result.correct, result.memory_tokens, interface.memory_budget, length_weight)
    result.calls = env.calls
    result.evidence_ids = list(env.evidence)
    result.answer_input_tokens = answer.last_input_tokens
    result.answer_output_tokens = answer.last_output_tokens
    return result
