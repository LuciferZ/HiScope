"""BM25 and dense cosine ranks with stable ID ties and reciprocal-rank fusion."""
from collections import Counter
import math
import re


def words(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class Ranker:
    def __init__(self, embed=None, k1=1.2, b=0.75, rrf_constant=60):
        self.embed, self.k1, self.b, self.rrf_constant = embed, k1, b, rrf_constant
        self.cache = {}

    def vector(self, text):
        if self.embed is None:
            raise ValueError("Dense/hybrid retrieval requires an embedding model")
        if text not in self.cache:
            raw = list(self.embed(text))[:768]
            norm = math.sqrt(sum(x*x for x in raw))
            self.cache[text] = [x / max(norm, 1e-12) for x in raw]
        return self.cache[text]

    def rank(self, query, documents, mode="hybrid", candidates=None):
        """BM25 statistics always use the full level corpus; filter only final scores."""
        ids = sorted(documents)
        allowed = set(ids if candidates is None else candidates)
        if not allowed <= documents.keys():
            raise ValueError("Unknown ranking candidate")
        if mode == "reasoning":
            return sorted(allowed)
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("Unknown rank mode")
        orders = []
        if mode in {"sparse", "hybrid"}:
            counts = {i: Counter(words(documents[i])) for i in ids}
            df = Counter(t for c in counts.values() for t in c)
            avg = sum(sum(c.values()) for c in counts.values()) / max(len(ids), 1)
            score = {}
            for i in allowed:
                c = counts[i]
                score[i] = sum(math.log(1 + (len(ids)-df[t]+0.5)/(df[t]+0.5)) *
                               c[t]*(self.k1+1) / (c[t]+self.k1*(1-self.b+self.b*sum(c.values())/max(avg, 1e-12)))
                               for t in words(query) if c[t])
            orders.append(sorted(allowed, key=lambda i: (-score[i], i)))
        if mode in {"dense", "hybrid"}:
            q = self.vector(query)
            score = {i: sum(a*b for a, b in zip(q, self.vector(documents[i]))) for i in allowed}
            orders.append(sorted(allowed, key=lambda i: (-score[i], i)))
        if mode != "hybrid":
            return orders[0]
        scores = {i: 0.0 for i in allowed}
        for order in orders:
            for rank, i in enumerate(order, 1):
                scores[i] += 1 / (self.rrf_constant + rank)
        return sorted(allowed, key=lambda i: (-scores[i], i))
