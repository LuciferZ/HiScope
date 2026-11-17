"""Reproduce Appendix S2's executor trace entirely offline, without model downloads."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrmem.data import Repository, read_jsonl
from mrmem.environment import Environment
from mrmem.ranking import Ranker

root = Path(__file__).resolve().parents[1]
repo = Repository.load(root/"examples/repository.json")
trace = read_jsonl(root/"examples/teacher_traces.jsonl")[0]
env = Environment(repo, "How long can I walk outdoors on May 25?", Ranker(), lambda s: len(s.split()))
for action in trace["actions"]:
    print(json.dumps(env.step(json.dumps(action)), indent=2))
    print(env.memory)
print("Synthetic offline demonstration; whitespace counts are not paper tokenizer counts.")
