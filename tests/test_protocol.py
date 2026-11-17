from dataclasses import replace
import json
from pathlib import Path
import unittest

from mrmem.data import Repository, Question, locomo_split, validate_split, Parent, Record, Source
from mrmem.environment import Environment, Interface, conventional_action
from mrmem.preparation import resize_repository, segment_sources
from mrmem.ranking import Ranker
from mrmem.rewards import reward, exact_match, correctness
from mrmem.evaluation import paired_contrast, factorial_analysis, choose_validation_candidate

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return Repository.load(ROOT/"examples/repository.json")


def action(ids, granularity="coarse", expand=False, mode="reasoning", top_k=5):
    return json.dumps(dict(subspace_ids=ids, retrieval_mode=mode, granularity=granularity,
                           top_k=top_k, expand=expand, stop=not expand))


class ExecutorTests(unittest.TestCase):
    def env(self, **kwargs):
        return Environment(fixture(), "walk outdoors", Ranker(), len, Interface(**kwargs))

    def test_paper_two_stage_trace(self):
        env = self.env()
        self.assertNotIn("leaves", env.metadata())
        self.assertTrue(env.step(action(["limit", "pref"], expand=True))["valid"])
        self.assertEqual(set(env.refinable), {"r1", "r2"})
        self.assertTrue(env.step(action(["r2"], "fine"))["valid"])
        self.assertIn("20 minutes", env.memory)
        self.assertIn("Prefers quiet outdoor trips", env.memory)
        self.assertNotIn("Needs short walks", env.memory)
        self.assertEqual(env.calls, 2)
        self.assertTrue(env.done)

    def test_scope_union_dedup_and_cap(self):
        env = self.env(protocol="scope", ranking="reasoning", top_k=1)
        result = env.step('{"subspace_ids":["limit","r2","pref"]}')
        self.assertEqual(result["resolved"], 2)
        self.assertEqual(len(env.evidence), 1)

    def test_parent_hidden_in_flat(self):
        env = self.env(view="flat", protocol="scope")
        self.assertFalse(env.step('{"subspace_ids":["limit"]}')["valid"])
        self.assertEqual(env.calls, 1)
        self.assertEqual(env.memory, "")

    def test_fine_overflow_keeps_coarse_and_other_records(self):
        env = self.env(memory_budget=100)
        env.step(action(["pref", "limit"], expand=True))
        old = env.memory
        self.assertLessEqual(len(old), 100)
        env.step(action(["r2"], "fine"))
        self.assertEqual(env.memory, old)

    def test_invalid_second_call_does_not_erase_evidence(self):
        env = self.env()
        env.step(action(["limit"], expand=True))
        old = env.memory
        self.assertFalse(env.step("not json")["valid"])
        self.assertEqual(env.memory, old)
        self.assertEqual(env.calls, 2)

    def test_no_unseen_leaf_first_call(self):
        env = self.env()
        self.assertFalse(env.step(action(["r2"]))["valid"])

    def test_refinement_descendants_before_record_cap(self):
        repo = fixture()
        r3 = replace(repo.records[1], id="r3", summary="another dated event")
        repo = Repository(repo.owner_id, repo.records+[r3],
            [repo.parents[0], replace(repo.parents[1], children=["r2", "r3"])], repo.sources)
        env = Environment(repo, "walk", Ranker(), len)
        env.step(action(["limit"], expand=True, top_k=1))
        self.assertEqual(env.refinable, {"r2", "r3"})
        self.assertTrue(env.step(action(["r3"], "fine"))["valid"])

    def test_strict_json_types_and_extra_fields(self):
        for change in ({"top_k": True}, {"expand": 1}, {"subspace_ids": "limit"}, {"unknown": 1}):
            data = json.loads(action(["limit"]))
            data.update(change)
            self.assertFalse(self.env().step(json.dumps(data))["valid"])

    def test_opaque_control_has_no_parent_semantics(self):
        env = self.env(protocol="scope", view="opaque", ranking="reasoning")
        for parent in env.metadata()["parents"]:
            self.assertTrue(parent["id"].startswith("node_"))
            self.assertEqual(parent["summary"], "(summary masked)")
        self.assertTrue(env.step('{"subspace_ids":["node_001"]}')["valid"])

    def test_hierarchical_leaf_only(self):
        env = self.env(protocol="scope", view="leaf_only", ranking="reasoning")
        self.assertIn("parents", env.metadata())
        self.assertFalse(env.step('{"subspace_ids":["limit"]}')["valid"])
        env = self.env(protocol="scope", view="leaf_only", ranking="reasoning")
        self.assertTrue(env.step('{"subspace_ids":["r2"]}')["valid"])

    def test_terminal_call_cannot_continue(self):
        env = self.env()
        env.step(action(["limit"]))
        with self.assertRaises(ValueError):
            env.step(action(["limit"]))

    def test_matched_granularity_experiment_locks_ranking(self):
        env = self.env(lock_executor=True)
        self.assertFalse(env.step(action(["limit"], mode="reasoning"))["valid"])
        env = self.env(lock_executor=True, ranking="reasoning")
        self.assertTrue(env.step(action(["limit"], mode="reasoning"))["valid"])


class DataAndMetricsTests(unittest.TestCase):
    def test_training_keeps_category_five_evaluation_excludes(self):
        q = Question("q", "c", "locomo", "question", "gold", "5", True)
        self.assertFalse(q.evaluable)
        self.assertEqual(correctness(q, "gold", training=True), 1)
        self.assertEqual(correctness(q, "gold", judge=lambda *a: 0, training=False), 0)

    def test_reward_gates_and_empty_memory(self):
        self.assertEqual(reward(0, 1), 0)
        self.assertEqual(reward(1, 0), 1)
        self.assertEqual(reward(1, 2048), 1)
        self.assertAlmostEqual(reward(1, 1024), 1.35)
        self.assertEqual(exact_match("The Cafe!", "cafe"), 1)
        self.assertEqual(exact_match("A cat, dog", "dog, cat", True), 1)

    def test_personamem_option_a_is_not_removed_as_article(self):
        q = Question("q", "u", "personamem", "Which option?", "A", "1", True)
        self.assertEqual(correctness(q, "", training=False), 0)
        self.assertEqual(correctness(q, "A.", training=False), 1)
        self.assertEqual(correctness(q, "B", training=True), 0)

    def test_folds_are_disjoint(self):
        tested = []
        for fold in range(10):
            split = locomo_split(fold)
            validate_split(split)
            self.assertEqual(len(split["train"]), 8)
            tested += split["test"]
        self.assertEqual(len(set(tested)), 10)

    def test_protected_standardization(self):
        repo, manifest = resize_repository(fixture(), size=1, protected_ids=["r2"])
        self.assertEqual(set(repo.leaves), {"r2"})
        self.assertEqual(manifest["removed"], ["r1"])
        with self.assertRaises(ValueError):
            resize_repository(fixture(), size=1, protected_ids=["r1", "r2"])
        with self.assertRaises(ValueError):
            resize_repository(fixture(), size=1)

    def test_donor_sources_namespaced_and_authorized(self):
        donor = replace(fixture(), owner_id="donor-user")
        with self.assertRaises(ValueError):
            resize_repository(fixture(), size=3, donors=[donor])
        repo, _ = resize_repository(fixture(), size=3, donors=[donor], allowed_donor_owners=[donor.owner_id])
        self.assertEqual(len(repo.records), 3)
        self.assertEqual(len(repo.parents), 2)
        self.assertTrue(any(i.startswith("donor:") for i in repo.leaves))

    def test_segmentation_thresholds(self):
        sources = [Source("a", "2025-01-01T00:00:00", "a"),
                   Source("b", "2025-01-02T00:00:00", "a"),
                   Source("c", "2025-01-03T00:01:00", "a")]
        segments = segment_sources(sources, lambda _: [1.0, 0.0])
        self.assertEqual([len(s) for s in segments], [2, 1])

    def test_dense_rrf_stable_ties(self):
        ranker = Ranker(lambda _: [1., 0.])
        self.assertEqual(ranker.rank("q", {"b": "x", "a": "x"}, "hybrid"), ["a", "b"])

    def test_bootstrap_qa_weights_and_seed_pairing(self):
        rows = []
        for owner, n, val in [("a", 1, 1), ("b", 2, 0), ("c", 3, 0)]:
            for q in range(n):
                for seed in [2027, 2028, 2029]:
                    for cell, score in [("A", val), ("B", 0)]:
                        rows.append(dict(owner_id=owner, question_id=str(q), seed=seed,
                                         condition=cell, correct=score))
        result = paired_contrast(rows, {"A": 1, "B": -1}, resamples=100)
        self.assertAlmostEqual(result["estimate"], 1/6)
        with self.assertRaises(ValueError):
            paired_contrast(rows[:-1], {"A": 1, "B": -1})

    def test_sft_rl_identity(self):
        rows = [dict(owner_id=str(o), question_id="q", seed=2027, condition=cell, correct=value)
                for o in range(3) for cell, value in
                {"FC": .4, "HC": .5, "FL": .6, "HL": .8, "FL_SFT": .5, "HL_SFT": .65}.items()]
        results = factorial_analysis(rows)
        self.assertAlmostEqual(results["interaction"]["estimate"], .1)
        self.assertAlmostEqual(results["sft_interaction"]["estimate"]+results["rl_increment"]["estimate"], .1)

    def test_two_history_interval_descriptive(self):
        rows = [dict(owner_id=o, question_id="q", seed=2027, condition="A", correct=1) for o in ["a", "b"]]
        self.assertIsNone(paired_contrast(rows, {"A": 1})["ci95"])

    def test_validation_tolerance_tiebreak(self):
        winner = choose_validation_candidate([dict(accuracy=.9, memory_tokens=100),
                                               dict(accuracy=.895, memory_tokens=50),
                                               dict(accuracy=.8, memory_tokens=1)])
        self.assertEqual(winner["memory_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
