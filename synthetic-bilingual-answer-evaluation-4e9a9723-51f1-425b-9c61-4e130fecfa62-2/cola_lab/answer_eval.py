"""Small bilingual, synthetic answer and evidence evaluation; no model download."""

import hashlib
import json
import platform
import re
from collections import Counter
from pathlib import Path


ENTITIES = {"validation": ("Alder", "Birch", "Cedar", "Dune"),
            "test": ("Ember", "Fjord", "Grove", "Harbor")}
VALUES = {"en": {"color": ("blue", "green", "red", "gold"),
                 "shape": ("round", "square", "oval", "flat")},
          "he": {"color": ("כחול", "ירוק", "אדום", "זהוב"),
                 "shape": ("עגול", "מרובע", "סגלגל", "שטוח")}}


def make_dataset():
    """Construct paired questions from fictional facts; no external data license needed."""
    documents, cases = [], []
    for split, entities in ENTITIES.items():
        for language in ("en", "he"):
            for index, entity in enumerate(entities):
                for attribute in ("color", "shape"):
                    answer = VALUES[language][attribute][index]
                    doc_id = f"{split}-{language}-{entity}-{attribute}"
                    statement = (f"The {attribute} of {entity} is {answer}." if language == "en"
                                 else f"{'הצבע' if attribute == 'color' else 'הצורה'} של {entity} הוא {answer}.")
                    documents.append({"id": doc_id, "split": split, "language": language,
                                      "entity": entity, "text": statement, "answer": answer})
                    # A paraphrase deliberately tests a lexical method's limits.
                    paraphrase = index == 0 and attribute == "color"
                    question = (f"Which hue does {entity} have?" if paraphrase and language == "en"
                                else f"איזה גוון יש ל {entity}?" if paraphrase
                                else f"What is the {attribute} of {entity}?" if language == "en"
                                else f"מה {'הצבע' if attribute == 'color' else 'הצורה'} של {entity}?")
                    cases.append({"id": f"{doc_id}-q", "split": split, "language": language,
                                  "entity": entity, "question": question,
                                  "expected_answer": answer, "evidence_ids": [doc_id]})
                question = (f"What is the weight of {entity}?" if language == "en"
                            else f"מה המשקל של {entity}?")
                cases.append({"id": f"{split}-{language}-{entity}-weight-q", "split": split,
                              "language": language, "entity": entity, "question": question,
                              "expected_answer": None, "evidence_ids": []})
    return {"source": "deterministic fictional facts generated in code", "documents": documents,
            "cases": cases}


def validate_dataset(dataset):
    if set(dataset) != {"source", "documents", "cases"} or not dataset["documents"] or not dataset["cases"]:
        raise ValueError("Invalid or empty dataset")
    docs, case_ids, entity_splits = {}, set(), {}
    for doc in dataset["documents"]:
        if set(doc) != {"id", "split", "language", "entity", "text", "answer"} or any(
                not isinstance(value, str) or not value.strip() for value in doc.values()):
            raise ValueError("Invalid document or missing value")
        if doc["id"] in docs or doc["split"] not in ENTITIES or doc["language"] not in VALUES:
            raise ValueError("Duplicate document or invalid split/language")
        docs[doc["id"]] = doc
        if doc["entity"] in entity_splits and entity_splits[doc["entity"]] != doc["split"]:
            raise ValueError("Entity leakage across splits")
        entity_splits[doc["entity"]] = doc["split"]
    counts = Counter()
    for case in dataset["cases"]:
        if set(case) != {"id", "split", "language", "entity", "question",
                         "expected_answer", "evidence_ids"}:
            raise ValueError("Invalid case fields")
        if case["id"] in case_ids or any(not isinstance(case[key], str) or not case[key].strip()
                                          for key in ("id", "split", "language", "entity", "question")):
            raise ValueError("Duplicate case or missing value")
        case_ids.add(case["id"])
        if case["split"] not in ENTITIES or case["language"] not in VALUES or not isinstance(case["evidence_ids"], list):
            raise ValueError("Invalid case split, language, or evidence")
        if entity_splits.get(case["entity"]) != case["split"]:
            raise ValueError("Entity leakage or unknown entity")
        answer = case["expected_answer"]
        if answer is None:
            if case["evidence_ids"]:
                raise ValueError("Unanswerable case has evidence")
        elif not isinstance(answer, str) or not answer.strip() or not case["evidence_ids"]:
            raise ValueError("Answerable case lacks answer or evidence")
        for doc_id in case["evidence_ids"]:
            doc = docs.get(doc_id)
            if doc is None or (doc["split"], doc["language"], doc["entity"], doc["answer"]) != (
                    case["split"], case["language"], case["entity"], answer):
                raise ValueError("Evidence does not support the expected answer")
        counts[case["split"], case["language"], answer is None] += 1
    if any(counts[split, language, unanswered] == 0 for split in ENTITIES
           for language in VALUES for unanswered in (False, True)):
        raise ValueError("Each split and language needs both case types")


def tokens(text):
    stopwords = {"what", "is", "the", "of", "which", "does", "have", "מה", "של", "הוא", "איזה", "יש", "ל"}
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE)) - stopwords


def retrieve(case, documents):
    """Rank same-split, same-language documents by lexical word overlap."""
    query = tokens(case["question"])
    candidates = [doc for doc in documents if doc["split"] == case["split"]
                  and doc["language"] == case["language"]]
    if not candidates:
        raise ValueError("No documents for case")
    return max(candidates, key=lambda doc: (len(query & tokens(doc["text"])), doc["id"])), max(
        len(query & tokens(doc["text"])) for doc in candidates)


def predict(case, documents, threshold):
    doc, overlap = retrieve(case, documents)
    if overlap < threshold:
        return {"answer": None, "citations": [], "overlap": overlap}
    return {"answer": doc["answer"], "citations": [doc["id"]], "overlap": overlap}


def score_case(case, prediction):
    """Exact answer plus required citation; abstention is correct only when expected."""
    expected = case["expected_answer"]
    answer = prediction["answer"]
    if expected is None:
        error = None if answer is None and not prediction["citations"] else "false_answer"
    elif answer is None:
        error = "false_abstention"
    elif answer.casefold().strip() != expected.casefold().strip():
        error = "wrong_answer"
    elif not set(case["evidence_ids"]).issubset(prediction["citations"]):
        error = "unsupported_citation"
    else:
        error = None
    return {"correct": error is None, "error_type": error}


def summarize(cases, predictions):
    scored = [score_case(case, prediction) for case, prediction in zip(cases, predictions)]
    answerable = [i for i, case in enumerate(cases) if case["expected_answer"] is not None]
    unanswerable = [i for i, case in enumerate(cases) if case["expected_answer"] is None]
    return {"count": len(cases), "accuracy": sum(item["correct"] for item in scored) / len(cases),
            "grounded_answer_rate": sum(scored[i]["correct"] for i in answerable) / len(answerable),
            "abstention_accuracy": sum(scored[i]["correct"] for i in unanswerable) / len(unanswerable),
            "error_counts": dict(Counter(item["error_type"] for item in scored if item["error_type"])),
            "errors": [{"case_id": case["id"], "language": case["language"],
                        "question": case["question"], "expected_answer": case["expected_answer"],
                        "expected_evidence": case["evidence_ids"], "prediction": predictions[i],
                        "error_type": scored[i]["error_type"]}
                       for i, case in enumerate(cases) if not scored[i]["correct"]]}


def evaluate(dataset, dataset_sha256):
    validate_dataset(dataset)
    docs = dataset["documents"]
    validation = [case for case in dataset["cases"] if case["split"] == "validation"]
    test = [case for case in dataset["cases"] if case["split"] == "test"]
    max_overlap = max(retrieve(case, docs)[1] for case in validation)
    candidates = range(max_overlap + 2)
    threshold = max(candidates, key=lambda value: (summarize(
        validation, [predict(case, docs, value) for case in validation])["accuracy"], value))
    methods = {}
    for method in ("always_abstain", "lexical_retrieval"):
        predictions = ([{"answer": None, "citations": [], "overlap": None} for _ in test]
                       if method == "always_abstain" else [predict(case, docs, threshold) for case in test])
        methods[method] = {"overall": summarize(test, predictions),
                           "by_language": {language: summarize(
                               [case for case in test if case["language"] == language],
                               [prediction for case, prediction in zip(test, predictions)
                                if case["language"] == language]) for language in VALUES}}
    return {"protocol": {"source": dataset["source"], "dataset_sha256": dataset_sha256,
                         "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "python": platform.python_version(), "languages": list(VALUES),
                         "selection": "one overlap threshold maximizes validation exact-answer-and-evidence accuracy; highest threshold wins ties",
                         "threshold": threshold, "test_use": "held out until threshold fixed",
                         "case_counts": {split: sum(case["split"] == split for case in dataset["cases"])
                                         for split in ENTITIES}},
            "methods": methods,
            "limitations": ["Synthetic paired templates are not natural user questions.",
                            "Document answer fields make this a rule-based retrieval baseline, not an LLM.",
                            "One small entity-disjoint split cannot establish language parity or stability.",
                            "Lexical overlap cannot resolve paraphrases or assess semantic equivalence."]}


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "answer_dataset.json"
    data_path.write_text(json.dumps(make_dataset(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw = data_path.read_bytes()
    dataset = json.loads(raw)
    report = evaluate(dataset, hashlib.sha256(raw).hexdigest())
    report_path = output / "answer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path
