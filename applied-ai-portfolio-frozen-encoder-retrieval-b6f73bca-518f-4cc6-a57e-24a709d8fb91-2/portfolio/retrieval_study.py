"""Data integrity and paired retrieval evaluation; usable without ML libraries.

The corpus is public to every arm. Query relevance labels are used for scoring
only. Topic groups cannot cross validation/test, including paraphrase siblings.
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .common import sha256_json

CANDIDATES = ("tfidf", "dense_mean", "dense_cls")
METRICS = ("precision_at_k", "recall_at_k", "mrr_at_k", "hit_at_k")
MAX_DOCUMENTS = 128
MAX_QUERIES = 128


@dataclass(frozen=True)
class Study:
    documents: tuple[dict, ...]
    queries: tuple[dict, ...]
    description: str

    def split(self, name):
        if name not in ("validation", "test"):
            raise ValueError("Unknown query split")
        return tuple(q for q in self.queries if q["split"] == name)

    def as_dict(self):
        return {"schema_version": 1, "description": self.description,
                "documents": list(self.documents), "queries": list(self.queries)}

    @property
    def fingerprint(self):
        return sha256_json(self.as_dict())


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value)


def validate_study(raw):
    if (not isinstance(raw, dict) or set(raw) != {"schema_version", "description", "documents", "queries"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise ValueError("Expected retrieval study schema_version 1")
    if not isinstance(raw["description"], str) or not raw["description"].strip():
        raise ValueError("Need a dataset description")
    docs, queries = raw["documents"], raw["queries"]
    if not isinstance(docs, list) or not 2 <= len(docs) <= MAX_DOCUMENTS:
        raise ValueError(f"Need 2..{MAX_DOCUMENTS} documents")
    if not isinstance(queries, list) or not 2 <= len(queries) <= MAX_QUERIES:
        raise ValueError(f"Need 2..{MAX_QUERIES} queries")
    ids, texts = set(), set()
    for rows, fields in ((docs, {"id", "group", "language", "text"}),
                         (queries, {"id", "group", "language", "text", "split", "relevant_ids"})):
        for row in rows:
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError("Unexpected document/query fields")
            if not _identifier(row["id"]) or not _identifier(row["group"]):
                raise ValueError("Invalid identifier or group")
            if row["id"] in ids:
                raise ValueError("Duplicate document/query ID")
            ids.add(row["id"])
            text = row["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                raise ValueError("Text must be nonblank and at most 4000 characters")
            key = normalized_text(text)
            if key in texts:
                raise ValueError("Duplicate normalized text or verbatim query/document copy")
            texts.add(key)
            if row["language"] not in ("en", "he"):
                raise ValueError("language must be en or he")
    by_id = {d["id"]: d for d in docs}
    split_groups = {"validation": set(), "test": set()}
    for q in queries:
        if not isinstance(q["split"], str) or q["split"] not in split_groups:
            raise ValueError("Query split must be validation or test")
        relevant = q["relevant_ids"]
        if (not isinstance(relevant, list) or not relevant
                or any(not isinstance(i, str) or i not in by_id for i in relevant)
                or len(set(relevant)) != len(relevant)):
            raise ValueError("Relevance labels must be unique, known document IDs")
        if any(by_id[i]["group"] != q["group"] for i in relevant):
            raise ValueError("Relevance labels must refer to the query topic group")
        split_groups[q["split"]].add(q["group"])
    if not all(split_groups.values()):
        raise ValueError("Both validation and test queries are required")
    if split_groups["validation"] & split_groups["test"]:
        raise ValueError("Topic-group leakage between validation and test")
    return Study(tuple(dict(d) for d in docs),
                 tuple({**q, "relevant_ids": list(q["relevant_ids"])} for q in queries),
                 raw["description"])


def load_study(path):
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Dataset exceeds 2 MB input budget")
    try:
        return validate_study(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset JSON at line {exc.lineno}") from exc


def check_k(k, document_count):
    if type(k) is not int or not 1 <= k <= min(10, document_count):
        raise ValueError("k must be an integer in [1, min(10, corpus size)]")


def score_ranking(ranked_ids, relevant_ids, k):
    """Denominator for precision is the requested k, even if an arm abstains."""
    if type(k) is not int or not 1 <= k <= 10:
        raise ValueError("Invalid k")
    if (not isinstance(ranked_ids, (list, tuple)) or not isinstance(relevant_ids, (list, tuple))
            or any(not isinstance(i, str) or not i for i in (*ranked_ids, *relevant_ids))
            or len(ranked_ids) > k or len(set(ranked_ids)) != len(ranked_ids)
            or not relevant_ids or len(set(relevant_ids)) != len(relevant_ids)):
        raise ValueError("Need unique ranked IDs and nonempty unique relevance labels")
    relevant = set(relevant_ids)
    positions = [i + 1 for i, item in enumerate(ranked_ids) if item in relevant]
    return {"precision_at_k": len(positions) / k,
            "recall_at_k": len(positions) / len(relevant),
            "mrr_at_k": 1 / positions[0] if positions else 0.0,
            "hit_at_k": float(bool(positions))}


def evaluate_queries(study, queries, retriever, k):
    check_k(k, len(study.documents))
    corpus = {d["id"]: d["text"] for d in study.documents}
    rows = []
    for q in queries:
        # Neither split metadata nor relevance labels are passed to a retriever.
        hits = retriever.search(q["text"], k=k)
        if not isinstance(hits, list) or len(hits) > k:
            raise ValueError("Retriever exceeded k or returned a malformed ranking")
        for hit in hits:
            if (not isinstance(hit, dict) or set(hit) != {"id", "text", "score"}
                    or not isinstance(hit["id"], str) or hit["id"] not in corpus
                    or hit["text"] != corpus[hit["id"]]
                    or type(hit["score"]) not in (int, float) or not math.isfinite(hit["score"])):
                raise ValueError("Retriever returned unknown, non-finite, or altered evidence")
        if any(a["score"] < b["score"] for a, b in zip(hits, hits[1:])):
            raise ValueError("Ranking must be sorted by descending score")
        metrics = score_ranking([h["id"] for h in hits], q["relevant_ids"], k)
        rows.append({"query_id": q["id"], "query": q["text"], "split": q["split"],
                     "group": q["group"], "language": q["language"],
                     "relevant_ids": q["relevant_ids"], "hits": hits, **metrics})
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    def aggregate(items):
        return {"queries": len(items), **{m: mean(r[m] for r in items) for m in METRICS}}
    return {**aggregate(rows), "by_language": {
        language: aggregate([r for r in rows if r["language"] == language])
        for language in sorted({r["language"] for r in rows})}}


def select_on_validation(candidate_rows):
    """No test scores accepted; ties use a fixed, predeclared candidate order."""
    if not candidate_rows or set(candidate_rows) - set(CANDIDATES):
        raise ValueError("Unknown or empty candidate set")
    paired_ids = None
    for rows in candidate_rows.values():
        if not rows or any(r["split"] != "validation" for r in rows):
            raise ValueError("Selection accepts validation rows only")
        ids = [r["query_id"] for r in rows]
        if len(set(ids)) != len(ids) or (paired_ids is not None and ids != paired_ids):
            raise ValueError("Candidates must use identical validation queries in the same order")
        paired_ids = ids
        for row in rows:
            if any(type(row[m]) not in (float, int) or not math.isfinite(row[m])
                   or not 0 <= row[m] <= 1 for m in METRICS):
                raise ValueError("Invalid validation metric")
    summaries = {name: summarize(rows) for name, rows in candidate_rows.items()}
    selected = max(summaries, key=lambda name: (
        summaries[name]["mrr_at_k"], summaries[name]["recall_at_k"], -CANDIDATES.index(name)))
    return {"selected": selected, "selection_split": "validation",
            "rule": "highest MRR@k, then Recall@k, then tfidf/dense_mean/dense_cls order",
            "validation": summaries}


def paired_differences(left, right):
    """Descriptive paired effects, averaged by topic; no independence claim."""
    if (not left or len(left) != len(right)
            or any((a["query_id"], a["group"], a["split"]) != (b["query_id"], b["group"], b["split"])
                   for a, b in zip(left, right))):
        raise ValueError("Need identical query/group/split pairs")
    groups = {}
    for a, b in zip(left, right):
        groups.setdefault(a["group"], []).append(b["mrr_at_k"] - a["mrr_at_k"])
    by_group = {g: mean(deltas) for g, deltas in sorted(groups.items())}
    return {"metric": "MRR@k", "direction": "alternative minus TF-IDF",
            "mean_query_difference": mean(b["mrr_at_k"] - a["mrr_at_k"] for a, b in zip(left, right)),
            "mean_group_difference": mean(by_group.values()), "by_group": by_group,
            "interpretation": "descriptive only; paraphrases within a topic are correlated"}


def compare(study, factories, k, *, on_selection=lambda decision: None):
    """Instantiate each arm on identical text-only Documents supplied by caller.

    Selection is persisted through the callback before any test query is searched.
    A failed arm aborts the comparison, rather than being replaced or omitted.
    """
    check_k(k, len(study.documents))
    if not factories or set(factories) - set(CANDIDATES) or "tfidf" not in factories:
        raise ValueError("Comparison requires TF-IDF and known candidate names")
    arms, validation, test, seconds = {}, {}, {}, {}
    for name in CANDIDATES:
        if name not in factories:
            continue
        start = time.perf_counter()
        arms[name] = factories[name]()
        validation[name] = evaluate_queries(study, study.split("validation"), arms[name], k)
        seconds[name] = {"index_and_validation": time.perf_counter() - start}
    decision = select_on_validation(validation)
    on_selection(decision)
    for name, arm in arms.items():
        start = time.perf_counter()
        test[name] = evaluate_queries(study, study.split("test"), arm, k)
        seconds[name]["test_search_and_scoring"] = time.perf_counter() - start
    return {"selection": decision, "validation_rows": validation, "test_rows": test,
            "test": {n: summarize(rows) for n, rows in test.items()}, "seconds": seconds,
            "paired_test_differences": {n: paired_differences(test["tfidf"], rows)
                                        for n, rows in test.items() if n != "tfidf"}}
