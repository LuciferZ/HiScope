"""Validated, model-independent data contracts and leakage-safe partitions."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import random


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Source:
    id: str
    timestamp: str
    text: str


@dataclass(frozen=True)
class Record:
    id: str
    category: str
    topic: str
    attribute: str
    value: str
    summary: str
    source_ids: list[str]

    @property
    def key(self):
        return self.category, self.topic, self.attribute

    def metadata(self):
        return {"id": self.id, "M": self.category, "D": self.topic,
                "L": self.attribute, "V": self.value}


@dataclass(frozen=True)
class Parent:
    id: str
    summary: str
    children: list[str]


@dataclass
class Repository:
    owner_id: str
    records: list[Record]
    parents: list[Parent]
    sources: list[Source]

    def __post_init__(self):
        self.leaves = {r.id: r for r in self.records}
        self.nodes = {p.id: p for p in self.parents}
        self.messages = {s.id: s for s in self.sources}
        if len(self.leaves) != len(self.records) or len(self.nodes) != len(self.parents):
            raise ValueError("Duplicate record/parent ID")
        if len(self.messages) != len(self.sources) or self.leaves.keys() & self.nodes.keys():
            raise ValueError("Duplicate source ID or overlapping parent/leaf IDs")
        children = []
        for parent in self.parents:
            if not parent.children or any(c not in self.leaves for c in parent.children):
                raise ValueError("Empty parent or missing child")
            if len({self.leaves[c].key for c in parent.children}) != 1:
                raise ValueError("A parent must represent one (M,D,L) tuple")
            children.extend(parent.children)
        if sorted(children) != sorted(self.leaves):
            raise ValueError("Each leaf must belong to exactly one parent")
        keys = [self.leaves[p.children[0]].key for p in self.parents]
        if len(keys) != len(set(keys)):
            raise ValueError("One parent per (M,D,L) tuple required")
        for r in self.records:
            if r.category not in {"episodic", "profile", "domain"}:
                raise ValueError("Unknown memory category")
            if not r.source_ids or any(s not in self.messages for s in r.source_ids):
                raise ValueError("Records require valid immutable source links")

    @classmethod
    def load(cls, path):
        return cls.from_dict(read_json(path))

    @classmethod
    def from_dict(cls, obj):
        return cls(obj["owner_id"], [Record(**r) for r in obj["records"]],
                   [Parent(**p) for p in obj["parents"]], [Source(**s) for s in obj["sources"]])

    def save(self, path):
        write_json(path, asdict(self))

    def resolve(self, ids):
        resolved = set()
        for item in ids:
            if item in self.nodes:
                resolved.update(self.nodes[item].children)
            elif item in self.leaves:
                resolved.add(item)
            else:
                raise ValueError(f"Unknown scope: {item}")
        return sorted(resolved)


@dataclass(frozen=True)
class Question:
    id: str
    owner_id: str
    dataset: str
    query: str
    answer: str
    category: str
    objective: bool
    list_answer: bool = False
    support_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if type(self.objective) is not bool or type(self.list_answer) is not bool:
            raise ValueError("objective/list_answer must be explicit booleans")
        if self.dataset not in {"locomo", "personamem", "synthetic"}:
            raise ValueError("Unsupported dataset")

    @property
    def evaluable(self):
        return not (self.dataset == "locomo" and str(self.category) == "5")


LOCOMO_IDS = ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44",
              "conv-47", "conv-48", "conv-49", "conv-50"]


def locomo_split(fold=None):
    if fold is None:
        return dict(train=LOCOMO_IDS[:6], dev=LOCOMO_IDS[6:8], test=LOCOMO_IDS[8:])
    if not 0 <= fold < 10:
        raise ValueError("fold must be 0..9")
    # Paper does not identify inner-dev assignment: fixed cyclic choice, recorded in manifest.
    test, dev = LOCOMO_IDS[fold], LOCOMO_IDS[(fold + 1) % 10]
    return dict(train=[x for x in LOCOMO_IDS if x not in {test, dev}], dev=[dev], test=[test])


def validate_split(split):
    for name in ("train", "dev", "test"):
        if not split[name] or len(split[name]) != len(set(split[name])):
            raise ValueError(f"Empty or duplicate {name} owners")
    if any(set(split[a]) & set(split[b]) for a, b in
           [("train", "dev"), ("train", "test"), ("dev", "test")]):
        raise ValueError("Owner leakage across partitions")


def personamem_split(owners, seed=2027):
    owners = sorted(set(owners))
    if len(owners) != 180:
        raise ValueError("Paper partition requires 180 users; provide a custom manifest otherwise")
    random.Random(seed).shuffle(owners)
    return dict(train=owners[:126], dev=owners[126:144], test=owners[144:])
