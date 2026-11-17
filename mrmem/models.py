"""Local Hugging Face adapters. No weights are downloaded unless explicitly allowed."""
from dataclasses import dataclass, field
import json
import os
import urllib.request


@dataclass
class ActionSample:
    prompt: str
    text: str
    prompt_ids: list[int]
    token_ids: list[int]
    old_log_prob: float = 0.0


@dataclass
class Trajectory:
    actions: list[ActionSample] = field(default_factory=list)
    reward: float = 0.0
    correct: int = 0
    memory_tokens: int = 0
    prediction: str = ""
    calls: int = 0
    gate_calls: int = 0
    invalid: int = 0
    policy_input_tokens: int = 0
    policy_output_tokens: int = 0
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    evidence_ids: list[str] = field(default_factory=list)


class HFPolicy:
    def __init__(self, path, device="cpu", dtype="float32", temperature=0.7,
                 local_only=True, checkpointing=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device, self.temperature = torch, device, temperature
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=getattr(torch, dtype), local_files_only=local_only).to(device)
        # Keep the probability model identical during rollout and optimization.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.eval()
        if checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            self.model.train()  # HF checkpointing requires training mode; dropout stays disabled
        self.limit = getattr(self.model.config, "max_position_embeddings", 32768)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_prompt(self, prompt):
        # Base checkpoints have no guaranteed chat template; fixed plain completion format.
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def freeze(self):
        self.model.requires_grad_(False)
        self.model.eval()
        return self

    def sample(self, prompt, max_tokens=256, greedy=False):
        torch = self.torch
        ids = self.encode_prompt(prompt)
        if len(ids) + max_tokens > self.limit:
            raise ValueError("Policy context overflow; metadata is never silently truncated")
        inputs = torch.tensor([ids], device=self.device)
        kwargs = dict(max_new_tokens=max_tokens, do_sample=not greedy,
                      pad_token_id=self.tokenizer.pad_token_id,
                      eos_token_id=self.tokenizer.eos_token_id,
                      repetition_penalty=1.0, no_repeat_ngram_size=0)
        if not greedy:
            kwargs.update(temperature=self.temperature, top_p=1.0, top_k=0)
        with torch.no_grad():
            output = self.model.generate(inputs, attention_mask=torch.ones_like(inputs), **kwargs)
        generated = output[0, len(ids):].tolist()
        return ActionSample(prompt, self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                            ids, generated)

    def action_logits(self, sample):
        torch = self.torch
        if not sample.prompt_ids or not sample.token_ids:
            raise ValueError("Empty prompt or action tokens")
        ids = torch.tensor([sample.prompt_ids + sample.token_ids], device=self.device)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        start = len(sample.prompt_ids) - 1
        return logits[0, start:start+len(sample.token_ids)].float() / self.temperature

    def score(self, sample, reference=None):
        """Sum action-token log probs and exact KL(ref||policy), including generated EOS."""
        torch = self.torch
        logp = self.action_logits(sample).log_softmax(-1)
        ids = torch.tensor(sample.token_ids, device=self.device)
        chosen = logp.gather(-1, ids[:, None]).sum()
        kl = chosen * 0.0
        if reference is not None:
            with torch.no_grad():
                ref = reference.action_logits(sample).log_softmax(-1).to(self.device)
            kl = (ref.exp() * (ref - logp)).sum()
        return chosen, kl

    def sft_loss(self, sample):
        # Supervised CE is untempered; sampling temperature applies only to the RL policy.
        import torch.nn.functional as F
        logits = self.action_logits(sample) * self.temperature
        targets = self.torch.tensor(sample.token_ids, device=self.device)
        return F.cross_entropy(logits, targets, reduction="sum"), len(sample.token_ids)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class HFAnswer:
    def __init__(self, path, device="cpu", dtype="float32", local_only=True):
        self.backend = HFPolicy(path, device, dtype, local_only=local_only).freeze()
        self.last_input_tokens = self.last_output_tokens = 0

    def count(self, text):
        return self.backend.count(text)

    def __call__(self, question, memory):
        content = ("Answer the question using the supplied memory. Be concise. If evidence is insufficient, "
                   "say so. For multiple-choice questions return only the option label.\n"
                   f"Memory:\n{memory}\nQuestion: {question}")
        tokenizer = self.backend.tokenizer
        if getattr(tokenizer, "chat_template", None):
            content = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            content += "\nAnswer:\n"
        sample = self.backend.sample(content, max_tokens=512, greedy=True)
        self.last_input_tokens = len(sample.prompt_ids)
        self.last_output_tokens = len(sample.token_ids)
        return sample.text


class HFEmbedding:
    def __init__(self, path, device="cpu", dtype="float32", local_only=True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(path, torch_dtype=getattr(torch, dtype),
                      local_files_only=local_only).to(device).eval().requires_grad_(False)

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False).to(self.device)
        with self.torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        # Single-item, right-padded input: last nonpadding token pooling for Qwen3 embeddings.
        index = int(inputs.attention_mask.sum().item()) - 1
        return hidden[0, index, :768].float().cpu().tolist()


class HTTPJudge:
    """OpenAI-compatible chat endpoint via stdlib; credentials only from environment."""
    def __init__(self, base_url, model="gpt-4-turbo", key_env="MRMEM_JUDGE_API_KEY"):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key_env = model, key_env

    def __call__(self, query, gold, prediction):
        prompt = ("Assess whether the prediction correctly answers the question according to the reference. "
                  "Accept equivalent wording, but reject contradictions or missing required facts. "
                  "Return only JSON: {\"correct\": true} or {\"correct\": false}.\n" +
                  json.dumps(dict(question=query, reference=gold, prediction=prediction), ensure_ascii=False))
        payload = dict(model=self.model, temperature=0, max_tokens=4096,
                       messages=[dict(role="user", content=prompt)])
        headers = {"Content-Type": "application/json"}
        if os.environ.get(self.key_env):
            headers["Authorization"] = "Bearer " + os.environ[self.key_env]
        request = urllib.request.Request(self.url, json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        value = json.loads(result["choices"][0]["message"]["content"])["correct"]
        if type(value) is not bool:
            raise ValueError("Judge returned an invalid correctness value")
        return int(value)
