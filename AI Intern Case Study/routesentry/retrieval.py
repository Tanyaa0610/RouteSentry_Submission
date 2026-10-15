"""Lightweight, local, free semantic retrieval over context_notes.csv.

This is the "AI-assisted" half of the pipeline: a from-scratch TF-IDF
cosine-similarity index (no embeddings API, no vector database -- with
only 10 notes a vector DB would be theatre, not engineering; see README
for the explicit "why not" discussion).

IMPORTANT: this module only RANKS candidates by textual similarity. It
never decides a verdict -- that is evidence.py's job, using deterministic
rules. Semantic similarity finds things that might be relevant; it is not
trusted to say whether they actually apply.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

from routesentry.models import Note
from routesentry.text_utils import tokenize


class TfidfIndex:
    def __init__(self, notes: List[Note]):
        self.notes = notes
        self._doc_tokens: Dict[str, List[str]] = {
            n.note_id: tokenize(f"{n.applies_to} {n.text}") for n in notes
        }
        self._idf = self._compute_idf()
        self._doc_vectors: Dict[str, Dict[str, float]] = {
            n.note_id: self._tfidf_vector(self._doc_tokens[n.note_id]) for n in notes
        }

    def _compute_idf(self) -> Dict[str, float]:
        n_docs = len(self.notes)
        df: Counter = Counter()
        for tokens in self._doc_tokens.values():
            for term in set(tokens):
                df[term] += 1
        # smoothed idf, always positive
        return {term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()}

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        vec = {term: (count / length) * self._idf.get(term, 0.0) for term, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def query(self, query_text: str, top_k: int) -> List[Tuple[Note, float]]:
        q_tokens = tokenize(query_text)
        q_vec = self._query_vector(q_tokens)
        scored = []
        for note in self.notes:
            d_vec = self._doc_vectors[note.note_id]
            sim = sum(q_vec.get(term, 0.0) * weight for term, weight in d_vec.items())
            scored.append((note, sim))
        scored.sort(key=lambda pair: (-pair[1], pair[0].note_id))
        return scored[:top_k]

    def _query_vector(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        vec = {term: (count / length) * self._idf.get(term, 0.0) for term, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}


def build_query(route: str, week_of) -> str:
    """Query text for a route-week: route name emphasised, plus generic
    cost-rise vocabulary so genuinely cost-impacting notes rank higher
    than merely-mentions-this-route notes."""
    origin, _, destination = route.partition("-")
    return (
        f"{origin} {destination} {route} {route} "
        "cost increase price rise surcharge disruption delay toll fuel diesel "
        "weather flood monsoon festival demand"
    )
