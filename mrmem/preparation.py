"""Fixed repository construction and auditable experimental standardization."""
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import random

from .data import Repository, Record, Parent, Source


def group_records(owner, records, sources, summarize=None):
    groups = defaultdict(list)
    for record in records:
        groups[record.key].append(record)
    parents = []
    for key, members in sorted(groups.items()):
        # Semantic IDs for the full hierarchy; opaque controls replace them in the view only.
        parent_id = "parent:" + "/".join(key)
        summaries = [r.summary for r in members]
        summary = summarize(key, summaries) if summarize else " | ".join(dict.fromkeys(summaries))
        parents.append(Parent(parent_id, summary, sorted(r.id for r in members)))
    used = {s for r in records for s in r.source_ids}
    return Repository(owner, records, parents, [s for s in sources if s.id in used])


def segment_sources(sources, embed, time_hours=24, topic_threshold=.75):
    segments = []
    previous = None
    for source in sorted(sources, key=lambda s: (datetime.fromisoformat(s.timestamp), s.id)):
        vector = embed(source.text)
        norm = sum(v*v for v in vector)**.5
        vector = [v/max(norm, 1e-12) for v in vector]
        now = datetime.fromisoformat(source.timestamp)
        boundary = previous is None
        if previous:
            old_time, old_vector = previous
            similarity = sum(a*b for a, b in zip(vector, old_vector))
            boundary = (now-old_time).total_seconds() > time_hours*3600 or similarity < topic_threshold
        if boundary:
            segments.append([])
        segments[-1].append(source)
        previous = now, vector
    return segments


def construct_repository(owner, sources, generator, embed):
    """A deterministic prompted extractor/consolidator. No query or answer is provided."""
    records = []
    for segment in segment_sources(sources, embed):
        prompt = ("Extract atomic memories from the source messages. Return only a JSON array. "
                  "Each object has category (episodic/profile/domain), topic, attribute, value, summary, source_ids. "
                  "Episodic: dated events/developments. Profile: personal attribute values and validity dates. "
                  "Domain: facts about distinct entities/concepts. Keep entities separate. "
                  "source_ids must cite supplied messages; summaries must preserve temporal qualifiers.\n" +
                  json.dumps([asdict(s) for s in segment], ensure_ascii=False))
        items = json.loads(generator(prompt))
        if not isinstance(items, list):
            raise ValueError("Extraction must return a JSON array")
        for item in items:
            if not set(item["source_ids"]) <= {s.id for s in segment}:
                raise ValueError("Extractor invented source links")
            record = Record(id=f"r{len(records):06d}", **item)
            # Candidate selection is fixed; the temperature-zero model checks equivalence.
            same = [r for r in records if r.category == record.category]
            if record.category == "episodic":
                target = embed(record.summary)
                tn = sum(x*x for x in target)**.5
                scored = []
                for r in same:
                    vec = embed(r.summary)
                    sim = sum(a*b for a, b in zip(target, vec))/max(tn*sum(x*x for x in vec)**.5, 1e-12)
                    if sim >= .80:
                        scored.append((sim, r))
                same = [r for _, r in sorted(scored, key=lambda p: (-p[0], p[1].id))[:10]]
            else:
                same = [r for r in same if r.key == record.key]
            if same:
                prompt = ("Decide whether the new record is the same event with a dated development (episodic), "
                          "same personal attribute (profile), or equivalent entity and attribute (domain). "
                          "Never merge distinct entities/events. Return only JSON {\"merge_id\": null} or "
                          "{\"merge_id\": \"existing_id\", \"value\": \"...\", \"summary\": \"...\"}. "
                          "On merge, retain all dated developments and historical states in summary; for profile "
                          "put the latest current value in value. Do not lose facts or temporal restrictions.\n" +
                          json.dumps(dict(new=asdict(record), candidates=[asdict(r) for r in same]), ensure_ascii=False))
                decision = json.loads(generator(prompt))
                if decision["merge_id"] is not None:
                    matches = [r for r in same if r.id == decision["merge_id"]]
                    if len(matches) != 1:
                        raise ValueError("Consolidator selected an unauthorized record")
                    old = matches[0]
                    records[records.index(old)] = replace(old, value=decision["value"], summary=decision["summary"],
                        source_ids=list(dict.fromkeys(old.source_ids + record.source_ids)))
                    continue
            records.append(record)
    def summarize(key, summaries):
        return generator("Summarize these descendant memories for routing. Preserve key facts and temporal scope. "
                         "Return only a concise summary, no JSON.\n" + json.dumps(dict(key=key, summaries=summaries)))
    return group_records(owner, records, sources, summarize)


def resize_repository(repo, size=500, seed=2027, protected_ids=None, donors=(),
                      allowed_donor_owners=(), random_subsample=False):
    """Caller supplies split-authorized donors; source IDs are namespaced on import."""
    rng = random.Random(seed)
    records, sources = list(repo.records), list(repo.sources)
    manifest = dict(owner_id=repo.owner_id, size=size, seed=seed, removed=[], added=[],
                    random_subsample=random_subsample)
    protected = set(protected_ids or [])
    if not protected <= repo.leaves.keys():
        raise ValueError("Protected evidence IDs not found in repository")
    if len(records) > size:
        if protected_ids is None and not random_subsample:
            raise ValueError("Evidence-preserving trimming requires explicit protected_ids (possibly empty)")
        eligible = [r.id for r in records if random_subsample or r.id not in protected]
        if len(eligible) < len(records)-size:
            raise ValueError("Cannot meet size without removing protected evidence")
        manifest["removed"] = rng.sample(sorted(eligible), len(records)-size)
        records = [r for r in records if r.id not in set(manifest["removed"])]
    elif len(records) < size:
        if any(d.owner_id not in allowed_donor_owners or d.owner_id == repo.owner_id for d in donors):
            raise ValueError("Donor violates explicit split/owner authorization")
        candidates = [(d, r) for d in donors for r in d.records]
        if len(candidates) < size-len(records):
            raise ValueError("Insufficient distinct donor records")
        for donor, record in rng.sample(candidates, size-len(records)):
            prefix = "donor:" + donor.owner_id + ":"
            new = replace(record, id=prefix+record.id, source_ids=[prefix+s for s in record.source_ids])
            records.append(new)
            sources.extend(replace(donor.messages[s], id=prefix+s) for s in record.source_ids)
            manifest["added"].append(dict(owner=donor.owner_id, original=record.id, imported=new.id))
    # Rebuild grouping; merge cross-benchmark distractors into matching existing (M,D,L) parents.
    sources = list({s.id: s for s in sources}.values())
    result = group_records(repo.owner_id, records, sources)
    manifest["repository_sha256"] = hashlib.sha256(json.dumps(asdict(result), sort_keys=True).encode()).hexdigest()
    return result, manifest
