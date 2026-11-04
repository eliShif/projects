# Professional research and engineering projects

Runnable projects, documented experiments and reproducible evaluation. Each project includes its purpose, implementation, technologies, run instructions and validation. Scientific limits and negative findings are retained in the individual reports.

## Current project index

<!-- job-pilot-project-index:start -->
| Project | Purpose |
|---|---|
| [applied-ai-portfolio](applied-ai-portfolio-3d51afe6-aaa9-4775-be90-9ab88dfe6669-3/) | Numeric service-policy questions require current evidence, correct arithmetic and a defensible decision to abstain. Retrieval alone does not ensure a correct answer, and valid calculations can still follow an incorrectly interpreted question. |
| [applied-ai-portfolio](applied-ai-portfolio-1f4ca8ae-96a1-4e54-b076-442bd3784e3e-1/) | Compact models and AI services require measured tradeoffs between task accuracy, latency, storage, memory and reliable execution. Smaller files or lower training loss can conceal slower inference, higher memory use or unchanged answer accuracy. Concurrent submissions and worker failures also require consistent job handling. |
| [Applied AI Portfolio — Frozen Encoder Retrieval](applied-ai-portfolio-frozen-encoder-retrieval-b6f73bca-518f-4cc6-a57e-24a709d8fb91-2/) | English and Hebrew paraphrases can have little word overlap with relevant documents. The project evaluates whether frozen sentence embeddings improve retrieval over TF-IDF under shared data, retrieval cutoffs and selection rules. |
| [Bilingual Answer Evaluation](bilingual-answer-evaluation-4e9a9723-51f1-425b-9c61-4e130fecfa62-3/) | Aggregate answer scores can conceal whether equivalent Hebrew and English questions received different evidence or produced different answers. This component makes paired outcomes, citations and required-source availability inspectable. |
| [Applied AI Security Triage](applied-ai-security-triage-560c6f69-15ba-4734-9255-94f335944472-1/) | Queued ML experiments can load changed weights, tokenizers or configurations after their protocol is frozen, undermining reproducibility. Security-triage workflows also need findings tied to scoped evidence, explicit abstention and recoverable execution. |
| [Synthetic Question Classification](synthetic-question-family-evaluation-d391be65-de05-40cc-97dc-691f1150a1ee-1/) | Evaluate whether lexical classifiers can assign synthetic English questions to eight task families for held-out service entities, while detecting entity leakage, wording sensitivity and train–validation performance gaps. |
| [Bilingual Answer Evaluation: Reference and Metrics](bilingual-answer-evaluation-reference-and-metric-validation-ba772b65-e5f6-422c-9e76-8a75299ef79d-3/) | The evaluator could accept source words that did not answer the queried entity or attribute, and could divide by zero when a subset lacked answerable or unanswerable cases. These defects weakened reference validation and metric interpretation. |
| [Hebrew–English Phenomenon Evaluation](bilingual-answer-evaluation-hebrew-phenomenon-slices-b6f73bca-518f-4cc6-a57e-24a709d8fb91-3/) | Aggregate bilingual scores can conceal whether Hebrew wording changes cause failures in source retrieval or answer extraction. This project examines possessive inflection, full/deficient spelling and Hebrew–English content-word substitutions using matched controls. |
<!-- job-pilot-project-index:end -->

## Reproducibility and history

Follow the README within each project. Fixed Git commit links preserve the exact versions used in earlier work. [Historical project variants and earlier evidence](HISTORY.md) remain available for reproducibility. This index highlights the current relevant implementations; each linked project has one technical-documentation entry point.
