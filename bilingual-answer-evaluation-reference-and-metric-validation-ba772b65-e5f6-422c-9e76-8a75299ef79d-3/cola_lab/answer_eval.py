"""Small bilingual, synthetic answer and evidence evaluation; no model download."""

import hashlib
import json
import platform
import re
from collections import Counter
from itertools import product
from pathlib import Path


ENTITIES = {"validation": ("Alder", "Birch", "Cedar", "Dune"),
            "test": ("Ember", "Fjord", "Grove", "Harbor")}
VALUES = {"en": {"color": ("blue", "green", "red", "gold"),
                 "shape": ("round", "square", "oval", "flat")},
          "he": {"color": ("כחול", "ירוק", "אדום", "זהוב"),
                 "shape": ("עגול", "מרובע", "סגלגל", "שטוח")}}
CONTEXT_MODES = ("shared-document", "all-documents")
RETRIEVAL_RULE = "word-overlap-v1; same split/language; highest document ID breaks ties"
REFERENCE_RULE = "synthetic-facts-v1; match question entity/attribute against all same-split/language source facts"
METRIC_POLICY = "conditional-rates-v2; undefined rates are null with case-type counts and reasons"

# These grammars describe the existing fictional corpus, not arbitrary language.
# They do not use VALUES, expected_answer, evidence_ids or the baseline answerer.
HEBREW_ATTRIBUTES = {"הצבע": "color", "הצורה": "shape", "המשקל": "weight"}
FACT_PATTERNS = {
    "en": r"The (?P<attribute>color|shape|weight) of (?P<entity>\w+) is (?P<value>\w+)\.",
    "he": r"(?P<attribute>הצבע|הצורה|המשקל) של (?P<entity>\w+) הוא (?P<value>\w+)\."}
QUESTION_PATTERNS = {
    "en": r"What is the (?P<attribute>color|shape|weight) of (?P<entity>\w+)\?",
    "he": r"מה (?P<attribute>הצבע|הצורה|המשקל) של (?P<entity>\w+)\?"}
HUE_PATTERNS = {"en": r"Which hue does (?P<entity>\w+) have\?",
                "he": r"איזה גוון יש ל (?P<entity>\w+)\?"}


def _source_fact(text, language):
    match = re.fullmatch(FACT_PATTERNS[language], text)
    if match is None:
        raise ValueError("Unsupported synthetic source template; cannot establish reference facts")
    attribute = match["attribute"]
    return (match["entity"], HEBREW_ATTRIBUTES.get(attribute, attribute), match["value"])


def _question_target(text, language):
    match = re.fullmatch(QUESTION_PATTERNS[language], text)
    if match is not None:
        attribute = match["attribute"]
        return match["entity"], HEBREW_ATTRIBUTES.get(attribute, attribute)
    match = re.fullmatch(HUE_PATTERNS[language], text)
    if match is not None:
        return match["entity"], "color"
    raise ValueError("Unsupported synthetic question template; cannot establish answerability")


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
    """Check gold answers and abstentions against parsed source facts.

    Absence is a closed-world claim about the complete split/language corpus,
    not the retrieved document. Every source must parse before absence can be
    asserted. This is bounded template validation, not independent annotation.
    """
    if (not isinstance(dataset, dict) or set(dataset) != {"source", "documents", "cases"}
            or not isinstance(dataset["source"], str) or not dataset["source"].strip()
            or any(not isinstance(dataset[key], list) or not dataset[key]
                   for key in ("documents", "cases"))):
        raise ValueError("Invalid or empty dataset")
    docs, facts, case_ids, entity_splits = {}, {}, set(), {}
    for doc in dataset["documents"]:
        if not isinstance(doc, dict) or set(doc) != {"id", "split", "language", "entity", "text", "answer"} or any(
                not isinstance(value, str) or not value.strip() for value in doc.values()):
            raise ValueError("Invalid document or missing value")
        if doc["id"] in docs or doc["split"] not in ENTITIES or doc["language"] not in VALUES:
            raise ValueError("Duplicate document or invalid split/language")
        if doc["entity"] in entity_splits and entity_splits[doc["entity"]] != doc["split"]:
            raise ValueError("Entity leakage across splits")
        entity_splits[doc["entity"]] = doc["split"]
        entity, attribute, value = _source_fact(doc["text"], doc["language"])
        if entity != doc["entity"]:
            raise ValueError(f"Source entity disagrees with document metadata: {doc['id']}")
        if value != doc["answer"]:
            raise ValueError(f"Document answer disagrees with source text: {doc['id']}")
        key = (doc["split"], doc["language"], entity, attribute)
        if key in facts:
            raise ValueError(f"Duplicate or conflicting source fact: {key}")
        facts[key] = (value, doc["id"])
        docs[doc["id"]] = doc
    counts = Counter()
    for case in dataset["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "split", "language", "entity", "question",
                                                     "expected_answer", "evidence_ids"}:
            raise ValueError("Invalid case fields")
        if any(not isinstance(case[key], str) or not case[key].strip()
               for key in ("id", "split", "language", "entity", "question")) or case["id"] in case_ids:
            raise ValueError("Duplicate case or missing value")
        case_ids.add(case["id"])
        evidence = case["evidence_ids"]
        if (case["split"] not in ENTITIES or case["language"] not in VALUES
                or not isinstance(evidence, list)
                or any(not isinstance(doc_id, str) or not doc_id for doc_id in evidence)
                or len(evidence) != len(set(evidence))):
            raise ValueError("Invalid case split, language, or evidence")
        if entity_splits.get(case["entity"]) != case["split"]:
            raise ValueError("Entity leakage or unknown entity")
        entity, attribute = _question_target(case["question"], case["language"])
        if entity != case["entity"]:
            raise ValueError(f"Question entity disagrees with case metadata: {case['id']}")
        if case["id"] != f"{case['split']}-{case['language']}-{entity}-{attribute}-q":
            raise ValueError(f"Case ID disagrees with question entity/attribute: {case['id']}")
        fact = facts.get((case["split"], case["language"], entity, attribute))
        answer = case["expected_answer"]
        if answer is None:
            if evidence:
                raise ValueError("Unanswerable case has evidence")
            if fact is not None:
                raise ValueError(f"Unanswerable label hides an available source fact: {case['id']}")
        else:
            if not isinstance(answer, str) or not answer.strip() or not evidence:
                raise ValueError("Answerable case lacks answer or evidence")
            if fact is None:
                raise ValueError(f"No source fact for the question's entity and attribute: {case['id']}")
            if answer != fact[0] or evidence != [fact[1]]:
                raise ValueError(f"Evidence does not support the question's entity and attribute: {case['id']}")
        counts[case["split"], case["language"], answer is None] += 1
    if any(counts[split, language, unanswered] == 0 for split in ENTITIES
           for language in VALUES for unanswered in (False, True)):
        raise ValueError("Each split and language needs both case types")
    # A missing translation would make the language comparison unpaired.
    pair_counts = Counter((case["split"], case["entity"], case["id"].rsplit("-", 2)[-2],
                           case["language"]) for case in dataset["cases"])
    pair_keys = {(split, entity, intent) for split, entity, intent, _ in pair_counts}
    if any(pair_counts[split, entity, intent, language] != 1
           for split, entity, intent in pair_keys for language in VALUES):
        raise ValueError("Bilingual cases must be paired by split, entity, and intent")


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


def pair_id(case):
    """A stable key shared by translations, independent of their list positions."""
    return f"{case['split']}/{case['entity']}/{case['id'].rsplit('-', 2)[-2]}"


def method_inputs(case, documents, context_mode="shared-document"):
    """Select evidence once, then expose only question, document IDs and source text.

    Retrieval sees the split/language pool; neither answerer sees gold labels or
    structured document answers. Even a low-overlap selection is passed through:
    abstention is an answerer decision, not a second retrieval rule.
    """
    if context_mode not in CONTEXT_MODES:
        raise ValueError("Unknown context mode")
    selected, _ = retrieve(case, documents)

    def public_input(docs):
        return {"question": case["question"],
                "documents": [{"id": doc["id"], "text": doc["text"]} for doc in docs]}

    model_docs = ([selected] if context_mode == "shared-document" else sorted(
        (doc for doc in documents if doc["split"] == case["split"]
         and doc["language"] == case["language"]), key=lambda doc: doc["id"]))
    return {"always_abstain": public_input([]),
            "lexical_retrieval": public_input([selected]),
            "local_model": public_input(model_docs)}


def input_fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def input_exposure(value):
    return {"document_ids": [doc["id"] for doc in value["documents"]],
            "input_sha256": input_fingerprint(value)}


def predict_from_input(value, threshold):
    """Extract the final word of a known synthetic fact template, without labels."""
    if len(value["documents"]) != 1:
        raise ValueError("Lexical answerer requires exactly one selected document")
    doc = value["documents"][0]
    overlap = len(tokens(value["question"]) & tokens(doc["text"]))
    match = re.fullmatch(r"(?:The (?:color|shape) of \w+ is|(?:הצבע|הצורה) של \w+ הוא) (\w+)\.",
                         doc["text"])
    if overlap < threshold or match is None:
        return {"answer": None, "citations": [], "overlap": overlap}
    return {"answer": match.group(1), "citations": [doc["id"]], "overlap": overlap}


def predict(case, documents, threshold):
    return predict_from_input(method_inputs(case, documents)["lexical_retrieval"], threshold)


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
    elif set(case["evidence_ids"]) != set(prediction["citations"]):
        error = "unsupported_citation"
    else:
        error = None
    return {"correct": error is None, "error_type": error}


def summarize(cases, predictions):
    """Score a nonempty subset; absent case types yield null, not a zero score."""
    if not cases or len(cases) != len(predictions):
        raise ValueError("Predictions must cover every case")
    scored = [score_case(case, prediction) for case, prediction in zip(cases, predictions)]
    answerable = [i for i, case in enumerate(cases) if case["expected_answer"] is not None]
    unanswerable = [i for i, case in enumerate(cases) if case["expected_answer"] is None]
    undefined = {}
    if not answerable:
        undefined["grounded_answer_rate"] = "No answerable cases in this subset."
    if not unanswerable:
        undefined["abstention_accuracy"] = "No unanswerable cases in this subset."
    return {"count": len(cases), "accuracy": sum(item["correct"] for item in scored) / len(cases),
            "answerable_count": len(answerable), "unanswerable_count": len(unanswerable),
            "grounded_answer_rate": (sum(scored[i]["correct"] for i in answerable) / len(answerable)
                                     if answerable else None),
            "abstention_accuracy": (sum(scored[i]["correct"] for i in unanswerable) / len(unanswerable)
                                    if unanswerable else None),
            "undefined_metrics": undefined,
            "error_counts": dict(Counter(item["error_type"] for item in scored if item["error_type"])),
            "errors": [{"case_id": case["id"], "language": case["language"],
                        "question": case["question"], "expected_answer": case["expected_answer"],
                        "expected_evidence": case["evidence_ids"], "prediction": predictions[i],
                        "error_type": scored[i]["error_type"]}
                       for i, case in enumerate(cases) if not scored[i]["correct"]]}


def paired_language_analysis(cases, predictions):
    """Count matched case outcomes without treating parallel templates as independent samples."""
    pairs = {}
    for case, prediction in zip(cases, predictions):
        key = (case["entity"], case["id"].rsplit("-", 2)[-2])
        pairs.setdefault(key, {})[case["language"]] = score_case(case, prediction)["correct"]
    if any(set(outcomes) != set(VALUES) for outcomes in pairs.values()):
        raise ValueError("Missing bilingual pair")
    counts = Counter((outcomes["en"], outcomes["he"]) for outcomes in pairs.values())
    return {"pair_count": len(pairs), "both_correct": counts[True, True],
            "both_wrong": counts[False, False], "english_only_correct": counts[True, False],
            "hebrew_only_correct": counts[False, True]}


def paired_case_audit(cases, method_predictions):
    """Expose matched outcomes and method disagreements for error inspection.

    These are observed error categories, not explanations of why a model failed.
    """
    if not cases or "lexical_retrieval" not in method_predictions or any(
            len(predictions) != len(cases) for predictions in method_predictions.values()):
        raise ValueError("Audit needs lexical retrieval and one prediction per case per method")
    pairs = {}
    for index, case in enumerate(cases):
        key = (case["entity"], case["id"].rsplit("-", 2)[-2])
        languages = pairs.setdefault(key, {})
        if case["language"] in languages:
            raise ValueError("Duplicate language in paired audit")
        outcomes = {}
        for method, predictions in method_predictions.items():
            prediction = predictions[index]
            result = score_case(case, prediction)
            outcomes[method] = {"correct": result["correct"],
                                "error_type": result["error_type"],
                                "answer": prediction["answer"],
                                "citations": prediction["citations"]}
        languages[case["language"]] = {"case_id": case["id"], "question": case["question"],
                                       "expected_answer": case["expected_answer"],
                                       "expected_evidence": case["evidence_ids"],
                                       "methods": outcomes}
    if any(set(languages) != set(VALUES) for languages in pairs.values()):
        raise ValueError("Audit needs complete English and Hebrew pairs")
    rows = []
    asymmetry = {method: Counter() for method in method_predictions}
    transitions = {method: Counter() for method in method_predictions if method != "lexical_retrieval"}
    for (entity, intent), languages in sorted(pairs.items()):
        for method in method_predictions:
            en = languages["en"]["methods"][method]["correct"]
            he = languages["he"]["methods"][method]["correct"]
            category = ("both_correct" if en and he else "both_wrong" if not en and not he
                        else "english_only_correct" if en else "hebrew_only_correct")
            asymmetry[method][category] += 1
        for language in VALUES:
            lexical = languages[language]["methods"]["lexical_retrieval"]["correct"]
            for method in transitions:
                candidate = languages[language]["methods"][method]["correct"]
                category = ("same_correct" if lexical and candidate else
                            "same_wrong" if not lexical and not candidate else
                            "improved" if candidate else "regressed")
                transitions[method][category] += 1
        rows.append({"entity": entity, "intent": intent, "languages": languages})
    return {"pair_count": len(rows), "pairs": rows,
            "outcome_counts": {method: dict(counts) for method, counts in asymmetry.items()},
            "versus_lexical": {method: dict(counts) for method, counts in transitions.items()}}


def describe_dataset(dataset):
    """Record observable data balance and split membership for audit."""
    return {split: {language: {
        "answerable": sum(case["expected_answer"] is not None for case in dataset["cases"]
                          if case["split"] == split and case["language"] == language),
        "unanswerable": sum(case["expected_answer"] is None for case in dataset["cases"]
                            if case["split"] == split and case["language"] == language),
        "entities": sorted({case["entity"] for case in dataset["cases"]
                            if case["split"] == split and case["language"] == language})
    } for language in VALUES} for split in ENTITIES}


def method_report(cases, predictions):
    result = {"overall": summarize(cases, predictions),
              "by_language": {language: summarize(
                  [case for case in cases if case["language"] == language],
                  [prediction for case, prediction in zip(cases, predictions)
                   if case["language"] == language]) for language in VALUES}}
    result["by_entity"] = {entity: summarize(
        [case for case in cases if case["entity"] == entity],
        [prediction for case, prediction in zip(cases, predictions)
         if case["entity"] == entity]) for entity in sorted({case["entity"] for case in cases})}
    # Only nonempty slices are summarized; an entirely empty input remains invalid.
    result["by_answerability"] = {}
    for name, missing in (("answerable", False), ("unanswerable", True)):
        indices = [i for i, case in enumerate(cases) if (case["expected_answer"] is None) == missing]
        if indices:
            result["by_answerability"][name] = summarize(
                [cases[i] for i in indices], [predictions[i] for i in indices])
    result["paired_languages"] = paired_language_analysis(cases, predictions)
    return result


def compare_methods(cases, reference_predictions, candidate_predictions):
    """Compare matched test cases; resample whole entities to retain bilingual pairing.

    The interval is descriptive for this tiny synthetic set, not a population CI.
    """
    if not cases or len(cases) != len(reference_predictions) or len(cases) != len(candidate_predictions):
        raise ValueError("Both methods must predict every case")
    entities = sorted({case["entity"] for case in cases})
    deltas = {}
    for entity in entities:
        indices = [i for i, case in enumerate(cases) if case["entity"] == entity]
        deltas[entity] = (sum(score_case(cases[i], candidate_predictions[i])["correct"] -
                              score_case(cases[i], reference_predictions[i])["correct"]
                              for i in indices) / len(indices))
    # The fixed study has four held-out entities: enumerate every cluster resample.
    if len(entities) != 4:
        raise ValueError("Comparison requires the four held-out entities")
    draws = sorted(sum(sample) / len(sample) for sample in product(deltas.values(), repeat=4))
    return {"accuracy_delta": sum(deltas.values()) / len(deltas),
            "entity_deltas": deltas,
            "cluster_resampling": {"unit": "entity", "draws": len(draws),
                                   "descriptive_95_percentile_interval":
                                   [draws[int(0.025 * (len(draws) - 1))],
                                    draws[int(0.975 * (len(draws) - 1))]],
                                   "includes_zero": draws[int(0.025 * (len(draws) - 1))] <= 0 <=
                                                    draws[int(0.975 * (len(draws) - 1))]}}


def load_model_predictions(path, dataset):
    """Read frozen, offline test outputs from one identified model run."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if set(payload) != {"model_id", "model_version", "predictions"} or any(
            not isinstance(payload[key], str) or not payload[key].strip()
            for key in ("model_id", "model_version")) or not isinstance(payload["predictions"], list):
        raise ValueError("Model output needs an identifier, version, and predictions list")
    test_cases = {case["id"]: case for case in dataset["cases"] if case["split"] == "test"}
    docs = {doc["id"]: doc for doc in dataset["documents"]}
    predictions = {}
    for item in payload["predictions"]:
        if not isinstance(item, dict) or set(item) != {"case_id", "answer", "citations"}:
            raise ValueError("Invalid model prediction fields")
        case_id = item["case_id"]
        if not isinstance(case_id, str) or case_id not in test_cases or case_id in predictions:
            raise ValueError("Unknown or duplicate test case ID")
        answer, citations = item["answer"], item["citations"]
        if answer is not None and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError("Answer must be text or null")
        if not isinstance(citations, list) or any(not isinstance(doc_id, str) for doc_id in citations) or len(citations) != len(set(citations)):
            raise ValueError("Citations must be a list of distinct IDs")
        if answer is None and citations:
            raise ValueError("Abstention cannot cite evidence")
        case = test_cases[case_id]
        if any(not isinstance(doc_id, str) or doc_id not in docs or
               docs[doc_id]["split"] != "test" or
               docs[doc_id]["language"] != case["language"] for doc_id in citations):
            raise ValueError("Citation references unknown or mismatched evidence")
        predictions[case_id] = {"answer": answer, "citations": citations}
    if set(predictions) != set(test_cases):
        raise ValueError("Model outputs must cover every test case exactly once")
    return payload["model_id"], payload["model_version"], hashlib.sha256(raw).hexdigest(), predictions


def evaluate(dataset, dataset_sha256, model_outputs=None):
    validate_dataset(dataset)
    docs = dataset["documents"]
    validation = [case for case in dataset["cases"] if case["split"] == "validation"]
    test = [case for case in dataset["cases"] if case["split"] == "test"]
    max_overlap = max(retrieve(case, docs)[1] for case in validation)
    candidates = range(max_overlap + 2)
    threshold = max(candidates, key=lambda value: (summarize(
        validation, [predict(case, docs, value) for case in validation])["accuracy"], value))
    methods = {}
    method_predictions = {}
    inputs = [method_inputs(case, docs) for case in test]
    for method in ("always_abstain", "lexical_retrieval"):
        predictions = ([{"answer": None, "citations": [], "overlap": None} for _ in test]
                       if method == "always_abstain" else [
                           predict_from_input(value["lexical_retrieval"], threshold) for value in inputs])
        methods[method] = method_report(test, predictions)
        method_predictions[method] = predictions
    model_protocol = None
    if model_outputs is not None:
        model_id, model_version, output_hash, by_id = model_outputs
        predictions = [by_id[case["id"]] for case in test]
        methods["local_model"] = method_report(test, predictions)
        method_predictions["local_model"] = predictions
        model_protocol = {"model_id": model_id, "model_version": model_version,
                          "output_sha256": output_hash,
                          "input": "saved test predictions; no model called by evaluator"}
    return {"protocol": {"source": dataset["source"], "dataset_sha256": dataset_sha256,
                         "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "python": platform.python_version(), "languages": list(VALUES),
                         "selection": "one overlap threshold maximizes validation exact-answer-and-evidence accuracy; highest threshold wins ties",
                         "threshold": threshold, "test_use": "held out until threshold fixed",
                         "case_counts": {split: sum(case["split"] == split for case in dataset["cases"])
                                         for split in ENTITIES},
                         "data_quality": describe_dataset(dataset),
                         "reference_validation": REFERENCE_RULE, "metric_policy": METRIC_POLICY,
                         "model_outputs": model_protocol,
                         "information_control": {
                             "status": "unverified_frozen_outputs" if model_outputs else "baseline_only",
                             "retrieval_rule": RETRIEVAL_RULE,
                             "case_exposures": [{"case_id": case["id"], "pair_id": pair_id(case),
                                 "methods": {**{method: input_exposure(value[method])
                                     for method in ("always_abstain", "lexical_retrieval")},
                                     **({"local_model": None} if model_outputs else {})}}
                                 for case, value in zip(test, inputs)]}},
            "methods": methods,
            "paired_audit": paired_case_audit(test, method_predictions),
            "comparisons": {method: compare_methods(test, method_predictions["always_abstain"], predictions)
                            for method, predictions in method_predictions.items()
                            if method != "always_abstain"},
            "comparisons_with_lexical": {
                "local_model": compare_methods(test, method_predictions["lexical_retrieval"],
                                                method_predictions["local_model"])
            } if model_outputs else {},
            "limitations": ["Synthetic paired templates are not natural user questions.",
                            "Reference validation parses a closed synthetic grammar; it is not independent annotation or general semantic validation.",
                            "Missing facts mean absent from the full split/language source corpus, not merely absent from retrieved evidence.",
                            "The baseline extracts answers from known synthetic source templates; it is not an LLM.",
                            "Imported predictions alone do not verify evidence exposure or generation settings.",
                            "Four test entities and their resampling interval cannot establish language parity or population stability.",
                            "Lexical overlap cannot resolve paraphrases or assess semantic equivalence."]}


def run(output_dir, predictions_path=None, *, dataset=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "answer_dataset.json"
    data_path.write_text(json.dumps(make_dataset() if dataset is None else dataset,
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw = data_path.read_bytes()
    dataset = json.loads(raw)
    model_outputs = load_model_predictions(predictions_path, dataset) if predictions_path else None
    report = evaluate(dataset, hashlib.sha256(raw).hexdigest(), model_outputs)
    report_path = output / "answer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report_path
