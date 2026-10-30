Original source: https://github.com/eliShif/projects/tree/26f91b8693546fc34ac99da0c01895004886ac02
Authorship is not inferred; this adaptation is AI-assisted.

The supplied adaptation checkpoint is pinned in the task metadata to revision
`88b12633d5bb49e95c01592b179b4ea7a76c9699`, project directory
`bilingual-answer-evaluation-auditable-gguf-runs-ba772b65-e5f6-422c-9e76-8a75299ef79d-3`.
This workspace already contained that evaluator, GGUF adapter, audit code, tests
and historical validation artifacts. They were inspected and reused locally;
no remote was accessed or changed. The supplied revision identifies the input,
not a claim that this new local extension was committed or published there.

Version 0.3.1 adds source-fact reference validation, explicit undefined metrics,
answerability slices and targeted regression tests. The existing journals,
protocol freezing, replay, retrieval, inference backend and older bigram
experiment are retained. New execution records are in `validation/reference-fix/`;
`validation/current/` and the other inherited logs describe earlier checkpoints.
