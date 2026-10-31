# Read-only security triage experiment

## Question and method

Can a low-rank policy update improve procedure selection while an evidence executor restricts findings and actions?
Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* (2023),
[arXiv:2302.12173](https://arxiv.org/abs/2302.12173), motivates treating retrieved strings as untrusted data.
This experiment maps that threat to hostile alert notes, fixed actor/time scope, an operation allowlist, and deterministic finding construction.
Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), [arXiv:2106.09685](https://arxiv.org/abs/2106.09685),
motivates freezing the base and fitting a low-rank update. The reference backend applies that algebra to a small classifier.

Backend: `reference_linear_policy_not_llm`. Selected step: 96 of 96.
Training and validation use separate actors/events; the selected checkpoint is fixed before held-out evaluation.
All three arms receive the same serialized, label-free observation and the same evidence executor.

## Observed held-out results

| Arm | Grounded correct / cases | Correct abstentions / expected | Failed plans/calls |
|---|---:|---:|---:|
| adapted | 24 / 24 | 9 / 9 | 0 |
| base | 6 / 24 | 0 / 9 | 18 |
| rule | 24 / 24 | 9 / 9 | 0 |

Paired changes: `{"improved": 18, "regressed": 0, "unchanged": 6}`.
Raw decisions, candidate scores, failures, citations and traces are retained in `runs.json`; `summary.json` contains family breakdowns.

## Limits and follow-up

- Synthetic event templates are shared across splits. Entity separation does not establish generalization to new attacks.
- The standard-library reference is a seeded linear classifier with a rank-2 update, not an LLM or a pretrained model. Its features omit note text by construction.
- The optional pretrained path uses actual local weights and PyTorch LoRA, with candidate answer-token NLL selection. It requires a separate explicit run; reference results do not verify that path.
- Candidate NLL is length-normalized and is not calibrated confidence. Constrained candidate selection does not evaluate free-form generation.
- The rule baseline is supplied the task policy and is a strong control. A learned policy may tie or underperform it; every such outcome is retained.
- Findings are temporal correlations. Maintenance intent, live threat detection, analyst productivity and production capacity were not measured.
- Hostile-note and invalid-plan checks cover these boundaries only; they do not establish general prompt-injection resistance.
- Collection completeness is an explicit caller assertion. Missing or falsified upstream events cannot be detected from a completeness flag.
- Single seed and small held-out set; paired counts are descriptive and no significance or operational improvement is claimed.
- Replay validates artifact consistency, not historical inference authenticity. Hashes are integrity checks, not signatures.
