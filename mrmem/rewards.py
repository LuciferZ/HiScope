"""Training-mask correctness and benchmark-specific evaluation remain distinct."""
import re
import string


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction, gold, list_answer=False):
    if list_answer:
        return int({normalize(x) for x in prediction.split(",")} ==
                   {normalize(x) for x in gold.split(",")})
    return int(normalize(prediction) == normalize(gold))


def correctness(question, prediction, judge=None, training=True):
    use_em = question.objective if training else question.dataset in {"personamem", "synthetic"}
    if use_em:
        if question.dataset == "personamem":
            # Option A is a label, not the English article 'a'. Empty output must never match it.
            predicted_label = prediction.strip().strip("()[] .:").casefold()
            gold_label = question.answer.strip().strip("()[] .:").casefold()
            return int(bool(predicted_label) and predicted_label == gold_label)
        return exact_match(prediction, question.answer, question.list_answer)
    if judge is None:
        raise ValueError("Non-objective training / LoCoMo evaluation requires a binary judge")
    score = judge(question.query, question.answer, prediction)
    if type(score) is not int or score not in (0, 1):
        raise ValueError("Judge must return integer 0 or 1")
    return score


def reward(correct, memory_tokens, budget=2048, length_weight=0.7, correctness_weight=1.0):
    if correct not in (0, 1) or memory_tokens < 0 or budget <= 0:
        raise ValueError("Invalid reward arguments")
    length = max(0.0, 1 - memory_tokens / budget) if memory_tokens else 0.0
    return correct * (correctness_weight + length_weight * length)
