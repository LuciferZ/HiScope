"""Six-field actions, metadata views, complete-unit budgets and two-call transitions."""
from dataclasses import dataclass, asdict
import json


@dataclass
class Interface:
    view: str = "hierarchical"  # flat, hierarchical, leaf_only, opaque, leaf_list
    protocol: str = "two_stage"  # two_stage, scope, fixed_coarse, fixed_fine, parent_fine
    granularity: str = "coarse"
    top_k: int = 5
    memory_budget: int = 2048
    action_budget: int = 256
    ranking: str = "hybrid"
    lock_executor: bool = False  # set for the matched-ranking granularity comparison

    def __post_init__(self):
        if self.view not in {"flat", "hierarchical", "leaf_only", "opaque", "leaf_list"}:
            raise ValueError("Unknown interface view")
        if self.protocol not in {"two_stage", "scope", "fixed_coarse", "fixed_fine", "parent_fine", "adaptive"}:
            raise ValueError("Unknown retrieval protocol")
        if self.granularity not in {"coarse", "fine"} or self.top_k < 1:
            raise ValueError("Invalid granularity/top_k")


@dataclass
class Action:
    subspace_ids: list[str]
    retrieval_mode: str
    granularity: str
    top_k: int
    expand: bool
    stop: bool

    def json(self):
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


class Environment:
    def __init__(self, repo, query, ranker, count_tokens, interface=None):
        self.repo, self.query, self.ranker, self.count = repo, query, ranker, count_tokens
        self.config = interface or Interface()
        self.calls, self.done = 0, False
        self.evidence = {}  # insertion order is acceptance order
        self.history, self.errors = [], []
        self.refinable = set()
        self.alias = {f"node_{i:03d}": p for i, p in enumerate(sorted(repo.nodes), 1)}
        self.inverse_alias = {v: k for k, v in self.alias.items()}
        self.leaf_docs = {r.id: json.dumps(r.metadata(), ensure_ascii=False) for r in repo.records}
        self.parent_docs = {p.id: p.summary for p in repo.parents}

    def metadata(self):
        c = self.config
        flat = c.view == "flat"
        show_leaves = flat or c.protocol in {"scope", "fixed_fine"} or self.calls > 0
        leaves = self.refinable if self.calls > 0 else self.repo.leaves.keys()
        result = {}
        if not flat:
            result["parents"] = []
            for p in sorted(self.repo.parents, key=lambda p: p.id):
                item = {"id": self.inverse_alias[p.id] if c.view == "opaque" else p.id,
                        "summary": "(summary masked)" if c.view == "opaque" else p.summary}
                if c.protocol in {"scope", "fixed_fine"}:
                    item["children"] = p.children
                result["parents"].append(item)
        if show_leaves:
            result["leaves"] = []
            for i in sorted(leaves):
                item = self.repo.leaves[i].metadata()
                if self.calls:
                    item["summary"] = self.repo.leaves[i].summary
                result["leaves"].append(item)
        return result

    def allowed_ids(self):
        metadata = self.metadata()
        ids = {x["id"] for x in metadata.get("leaves", [])}
        if self.calls == 0 and self.config.view not in {"flat", "leaf_only", "leaf_list"}:
            ids.update(x["id"] for x in metadata.get("parents", []))
        return ids

    def prompt(self):
        c = self.config
        if c.protocol == "scope":
            schema = '{"subspace_ids":["ID",...]}'
            rules = f"Select scopes only. Executor fixes {c.ranking}, top_k={c.top_k}, {c.granularity}."
            if c.view in {"flat", "leaf_only", "leaf_list"}:
                rules += " Only leaf IDs are legal actions; parent IDs, if shown, are routing information."
        else:
            schema = ('{"subspace_ids":["ID",...],"retrieval_mode":"dense|sparse|hybrid|reasoning",'
                      '"granularity":"coarse|fine","top_k":1,"expand":false,"stop":true}')
            rules = ("Use expand=true, stop=false, granularity=coarse to request one refinement call. "
                     "Otherwise use expand=false, stop=true. Refinement selects visible leaves at fine granularity. "
                     f"At most {c.top_k} records. Protocol={c.protocol}; ranking={c.ranking}.")
            if c.protocol != "two_stage":
                rules += " One call only; ranking and top_k are fixed by this experiment."
            if self.calls:
                rules += " Select only displayed leaf IDs for refinement."
            if c.lock_executor:
                rules += f" Use retrieval_mode={c.ranking} and top_k={c.top_k} exactly."
        return ("Select relevant memory. Return exactly one JSON object, without explanation.\n"
                f"Schema: {schema}\n{rules}\n"
                "Select only IDs visible in metadata and permitted by this protocol.\n"
                f"Question: {self.query}\nMetadata: {json.dumps(self.metadata(), ensure_ascii=False)}\n"
                f"Previous actions: {json.dumps(self.history, ensure_ascii=False)}\n"
                f"Accepted evidence:\n{self.memory}\nAction:\n")

    @property
    def memory(self):
        return "\n\n".join(unit["text"] for unit in self.evidence.values())

    def parse(self, raw):
        if self.count_action(raw) > self.config.action_budget:
            raise ValueError("Action exceeds token budget")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("Action must be a JSON object")
        c = self.config
        if c.protocol == "scope":
            if set(obj) != {"subspace_ids"}:
                raise ValueError("Scope-only action has only subspace_ids")
            obj = dict(obj, retrieval_mode=c.ranking, granularity=c.granularity,
                       top_k=c.top_k, expand=False, stop=True)
        if set(obj) != set(Action.__dataclass_fields__):
            raise ValueError("Action must contain exactly the six fields")
        a = Action(**obj)
        if not isinstance(a.subspace_ids, list) or any(not isinstance(i, str) for i in a.subspace_ids):
            raise ValueError("subspace_ids must be a list of strings")
        if not set(a.subspace_ids) <= self.allowed_ids():
            raise ValueError("Action names hidden, forbidden, or nonexistent IDs")
        if type(a.top_k) is not int or not 1 <= a.top_k <= c.top_k:
            raise ValueError("top_k outside configured range")
        if type(a.expand) is not bool or type(a.stop) is not bool or a.stop == a.expand:
            raise ValueError("Use exactly one of expand/stop")
        if a.retrieval_mode not in {"dense", "sparse", "hybrid", "reasoning"}:
            raise ValueError("Invalid retrieval mode")
        if a.granularity not in {"coarse", "fine"}:
            raise ValueError("Invalid granularity")
        if a.expand and (self.calls > 0 or c.protocol != "two_stage" or a.granularity != "coarse"):
            raise ValueError("Expansion requires first-call coarse evidence in two_stage")
        if self.calls and a.granularity != "fine":
            raise ValueError("Refinement must request fine evidence")
        if c.lock_executor and (a.retrieval_mode != c.ranking or a.top_k != c.top_k):
            raise ValueError("Action violates locked ranking/top_k experiment settings")
        if c.protocol in {"fixed_coarse", "fixed_fine", "parent_fine"}:
            granularity = "coarse" if c.protocol == "fixed_coarse" else "fine"
            if a.granularity != granularity or a.top_k != c.top_k or a.retrieval_mode != c.ranking:
                raise ValueError("Action violates fixed experimental settings")
        if c.view == "leaf_list":
            a.subspace_ids = a.subspace_ids[:8]
        return a

    def count_action(self, raw):
        return self.action_counter(raw) if hasattr(self, "action_counter") else self.count(raw)

    def _unit(self, record_id, granularity):
        record = self.repo.leaves[record_id]
        if granularity == "coarse":
            text = f"[{record_id}] {record.summary}"
        else:
            sources = [self.repo.messages[i] for i in dict.fromkeys(record.source_ids)]
            text = "\n".join(f"[{record_id}/{s.id} @ {s.timestamp}] {s.text}" for s in sources)
        return dict(text=text, granularity=granularity, source_ids=list(record.source_ids))

    def step(self, raw):
        if self.done:
            raise ValueError("Episode already terminated")
        try:
            action = self.parse(raw)
            ids = [self.alias.get(i, i) if self.config.view == "opaque" else i for i in action.subspace_ids]
            candidates = self.repo.resolve(ids)
            ranked = self.ranker.rank(self.query, self.leaf_docs, action.retrieval_mode, candidates)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.calls += 1
            self.history.append(raw)
            self.errors.append(str(error))
            self.done = True  # invalid first call cannot initiate expansion; invalid second retains coarse
            return {"valid": False, "error": str(error), "returned": []}
        self.calls += 1
        self.history.append(raw)
        returned = []
        for rid in ranked[:action.top_k]:
            unit = self._unit(rid, action.granularity)
            proposed = dict(self.evidence)
            proposed[rid] = unit
            text = "\n\n".join(u["text"] for u in proposed.values())
            if self.count(text) <= self.config.memory_budget:
                self.evidence = proposed
                returned.append(rid)
            # Oversized fine units do not erase the accepted coarse version.
        if action.expand:
            self.refinable = set(candidates)  # all descendant leaves, before top-k
        self.done = action.stop or self.calls >= 2
        return {"valid": True, "returned": returned, "resolved": len(candidates)}


def conventional_action(env, k_topic=3):
    if env.config.protocol != "scope":
        raise ValueError("Conventional factorial comparator uses scope-only protocol")
    if env.config.view == "flat":
        ids = list(env.repo.leaves)
    else:
        ids = env.ranker.rank(env.query, env.parent_docs, "hybrid")[:k_topic]
        if env.config.view == "opaque":
            ids = [env.inverse_alias[i] for i in ids]
        elif env.config.view in {"leaf_only", "leaf_list"}:
            ids = env.repo.resolve(ids)
    # Conventional ranker is not constrained by the learned 256-token scope encoding.
    return json.dumps({"subspace_ids": ids}, separators=(",", ":"))
