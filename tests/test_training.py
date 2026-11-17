import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from mrmem.data import Question, Repository, write_json
from mrmem.environment import Interface
from mrmem.models import ActionSample, HFPolicy
from mrmem.ranking import Ranker
from mrmem.training import TrainConfig, grpo_loss, group_advantages, train_grpo, train_sft


class TinyTokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"good": 0, "bad": 1, "eos": 2}

    def encode(self, text, add_special_tokens=False):
        return [0] if "r2" in text else [1]


class TinyPolicy:
    """Synthetic categorical policy for testing the real rollout/optimizer loop.

    Each branch emits one complete JSON scope action. This is not a paper backbone.
    """
    def __init__(self):
        self.torch = torch
        self.temperature = .7
        self.tokenizer = TinyTokenizer()
        self.limit = 1024
        self.model = torch.nn.Linear(1, 2, bias=False)
        torch.nn.init.zeros_(self.model.weight)

    def freeze(self):
        self.model.requires_grad_(False)
        return self

    def count(self, text):
        return len(text.split())

    def encode_prompt(self, prompt):
        return [0]

    def log_probs(self):
        return (self.model(torch.ones(1)).float()/self.temperature).log_softmax(-1)

    def sample(self, prompt, max_tokens=256):
        index = int(torch.multinomial(self.log_probs().exp(), 1))
        text = json.dumps({"subspace_ids": ["r2" if index == 0 else "r1"]})
        return ActionSample(prompt, text, [0], [index])

    def score(self, sample, reference=None):
        p = self.log_probs()
        kl = p.sum()*0
        if reference:
            ref = reference.log_probs().detach()
            kl = (ref.exp()*(ref-p)).sum()
        return p[sample.token_ids[0]], kl

    def sft_loss(self, sample):
        return -self.log_probs()[sample.token_ids[0]], 1

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), Path(path)/"tiny.pt")


class TinyAnswer:
    last_input_tokens = 1
    last_output_tokens = 1

    def count(self, text):
        return len(text.split())

    def __call__(self, question, memory):
        return "20 minutes" if "20 minutes" in memory else "unknown"


class TrainingTests(unittest.TestCase):
    def test_group_zero_variance_no_nan(self):
        advantages = group_advantages([[0, 0], [1, 2]])
        self.assertTrue(torch.equal(advantages[0], torch.zeros(2)))
        self.assertTrue(torch.allclose(advantages[1], torch.tensor([-1., 1.])))

    def test_trajectory_ratio_is_sum_not_token_average(self):
        logp = torch.tensor([.1, .2], requires_grad=True).sum()
        loss, diagnostics = grpo_loss(logp, 0., 1., logp*0, epsilon=.2, beta=0)
        self.assertAlmostEqual(loss.item(), -1.2, places=6)
        self.assertEqual(diagnostics["clip_fraction"], 1)

    def test_negative_advantage_clipping_direction(self):
        loss, _ = grpo_loss(torch.tensor(-1.), 0., -1., torch.tensor(0.), beta=0)
        self.assertAlmostEqual(loss.item(), .8, places=6)

    def test_kl_remains_active_for_zero_advantage(self):
        kl = torch.tensor(.7, requires_grad=True)
        loss, _ = grpo_loss(torch.tensor(0.), 0., 0., kl, beta=.01)
        loss.backward()
        self.assertAlmostEqual(kl.grad.item(), .01, places=6)

    def test_actual_grpo_updates_and_frozen_reference(self):
        repo = Repository.load(Path(__file__).resolve().parents[1]/"examples/repository.json")
        q = Question("q", repo.owner_id, "synthetic", "How long?", "20 minutes", "1", True)
        policy = TinyPolicy()
        reference = TinyPolicy()
        reference.model.load_state_dict(policy.model.state_dict())
        reference.freeze()
        original_reference = reference.model.weight.detach().clone()
        config = TrainConfig(updates=3, groups_per_batch=2, group_size=8, update_epochs=1,
                             minibatch_size=8, learning_rate=.1, warmup_updates=1)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"grpo"
            train_grpo(policy, reference, [q], {repo.owner_id: repo}, TinyAnswer(), Ranker(),
                       Interface(view="flat", protocol="scope", granularity="fine", ranking="reasoning"),
                       output, config)
            self.assertTrue((output/"last/tiny.pt").exists())
            self.assertEqual(len((output/"train.jsonl").read_text().splitlines()), 3)
        self.assertGreater(policy.log_probs()[0].exp().item(), .5)
        self.assertTrue(torch.equal(reference.model.weight, original_reference))

    def test_actual_sft_update_and_owner_leakage(self):
        policy = TinyPolicy()
        train = [dict(owner_id="train", prompt="question", completion='{"subspace_ids":["r2"]}')]
        dev = [dict(train[0], owner_id="dev")]
        with tempfile.TemporaryDirectory() as temp:
            train_sft(policy, train, dev, Path(temp)/"sft", epochs=2, learning_rate=.1)
            self.assertGreater(policy.log_probs()[0].exp().item(), .5)
            self.assertTrue((Path(temp)/"sft/best/tiny.pt").exists())
            with self.assertRaises(ValueError):
                train_sft(policy, train, train, Path(temp)/"leak")

    def test_hf_logits_alignment_and_kl_direction(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(1)
        policy = HFPolicy.__new__(HFPolicy)
        policy.torch, policy.device, policy.temperature = torch, "cpu", .7
        policy.model = GPT2LMHeadModel(GPT2Config(vocab_size=17, n_positions=32, n_embd=8,
            n_layer=1, n_head=1, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)).eval()
        ref = HFPolicy.__new__(HFPolicy)
        ref.torch, ref.device, ref.temperature = torch, "cpu", .7
        ref.model = copy.deepcopy(policy.model)
        ref.freeze()
        with torch.no_grad():
            policy.model.transformer.wte.weight[3].add_(.3)
        sample = ActionSample("", "", [1, 2], [3, 4, 5])
        chosen, kl = policy.score(sample, ref)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        current = (policy.model(ids).logits[0, 1:4].float()/.7).log_softmax(-1)
        reference = (ref.model(ids).logits[0, 1:4].float()/.7).log_softmax(-1)
        expected = sum(current[i, token] for i, token in enumerate([3, 4, 5]))
        self.assertTrue(torch.allclose(chosen, expected))
        self.assertTrue(torch.allclose(kl, (reference.exp()*(reference-current)).sum()))
        (chosen+kl).backward()
        self.assertIsNotNone(policy.model.transformer.wte.weight.grad)
        self.assertIsNone(ref.model.transformer.wte.weight.grad)

    def test_hf_local_loading_sampling_and_checkpoint_roundtrip(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "question": 3, "answer": 4, "memory": 5}
        raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]",
                                            eos_token="[EOS]", pad_token="[PAD]")
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=32, n_embd=8,
            n_layer=1, n_head=1, bos_token_id=1, eos_token_id=1, pad_token_id=2,
            resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/"source"
            model.save_pretrained(source)
            tokenizer.save_pretrained(source)
            policy = HFPolicy(str(source))
            sample = policy.sample("question memory", max_tokens=3)
            self.assertGreater(len(sample.token_ids), 0)
            score, _ = policy.score(sample)
            self.assertTrue(torch.isfinite(score))
            policy.save(Path(temp)/"saved")
            restored = HFPolicy(str(Path(temp)/"saved"))
            self.assertTrue(torch.allclose(policy.score(sample)[0], restored.score(sample)[0]))


if __name__ == "__main__":
    unittest.main()
