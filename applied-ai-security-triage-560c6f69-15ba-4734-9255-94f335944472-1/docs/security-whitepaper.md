# Draft technical note: bounded investigation before generated conclusions

Security logs can contain attacker-controlled text. Greshake et al.,
[*Not what you've signed up for: Compromising Real-World LLM-Integrated
Applications with Indirect Prompt Injection*](https://arxiv.org/abs/2302.12173)
(2023), describes how external content can cross a model's instruction boundary.
This prototype explores a narrow engineering response: allow a planner to choose
an investigation procedure while deterministic tools own evidence scope,
correlation, citations and the final finding.

An alert supplies an actor, anchor event, bounded time window and collection
completeness assertion. Retrieval cannot search another actor or silently discard
excess events. The existing versioned evidence executor supplies policy values;
conflicting current values cause abstention. The plan can request authentication
or access investigation, or abstain. Invalid actions and procedures incompatible
with the anchor are rejected. Findings describe temporal correlations and do not
assign malicious intent. No response action is exposed.

The experiment separates data preparation, adaptation and evaluation into durable
queue stages. A worker writes to an attempt-specific directory and publishes a
receipt only while holding its lease. Data and source digests, raw predictions,
failures and validation history accompany the result. Replay recomputes findings
and metrics from the recorded plan and evidence. It cannot certify that a model
historically produced a claimed response.

Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685)
(2021), supplies the low-rank update method. The offline experiment trains a
small linear reference policy with analytic gradients. An optional path uses the
repository's PyTorch LoRA layers and actual local pretrained causal-LM weights.
Both select checkpoints using validation NLL and evaluate a fixed held-out split
against a deterministic rule under identical information conditions.

Observed counts and failures are generated from the recorded rows in the
[experiment report](../results/security_triage_v2/REPORT.md). The reference run is
not evidence of LLM behavior: it uses structured features and does not consume
note text. The separate pretrained security experiment remains unrun in the
supplied evidence. ML dependencies and local model weights were unavailable in
the restricted builder environment on 31 October 2026 (Python 3.14.4).
The later worker installed CPU PyTorch `2.15.0+cpu` and passed the reference
security suite, wheel build and finite smoke. It supplied no successful optional
tensor-suite execution or pretrained security experiment; installation of the
full ML stack and availability of local pretrained weights are not established.

The resumed implementation exposed an additional data-integrity failure. A
planner receiving the executor's mutable dictionary could rewrite event fields
or policy values and affect the finding and subsequent comparison arm. Four
[regression tests failed before the fixes](../results/security_validation_v2/regressions_before_fix.txt).
The executor now retains its own snapshot and passes a separate copy to each
planner. JSON boundaries also reject duplicate keys instead of silently retaining
the last value. Evidence isolation is an aliasing control, not process isolation
for arbitrary Python plugins.

The new run trains 22 low-rank reference parameters for 96 updates. Its held-out
result is 24/24 grounded-correct outcomes, tying the deterministic rule; the
unadapted random policy produces 6/24 and 18 rejected plans. The learned policy
does not improve on the explicit rule control. This is a narrow procedure-selection
result, not a security detector or an operational-benefit estimate.

A separate acceptance record in the same evaluation receipt combines 64 HTTP
submissions for 32 jobs, four workers, seven rejected requests and recovery of
one expired lease after reopening SQLite. It also records rejection of five
adversarial planner fixtures and three stale lease operations. Audit recomputes
findings, checks traces and compares the report to saved counts. The HTTP handler
uses in-memory byte streams and the lease clock is injected; this does not
measure live network behavior, throughput or recovery time. Fixture outcomes are
kept separate from model evaluation rows.

The next research step is a prespecified pretrained-model experiment on varied
investigation language, independent event templates and multiple injection
placements. Current synthetic tests establish bounded implementation behavior;
they do not measure operational detection quality, analyst productivity, broad
injection robustness or deployment capacity. This note is a project draft, and
the generated experiment report is the record of observed results.
