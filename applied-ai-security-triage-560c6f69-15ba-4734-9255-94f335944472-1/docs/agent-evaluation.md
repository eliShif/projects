# Agent Evaluation and Adaptation

## Engineering question

Does retrieval or autonomous tool selection improve answers to unfamiliar service
policies? Can low-rank adaptation improve the format without changing base weights?
This project tests these questions; it does not assume that an agent beats a
simpler workflow or that lower training loss implies better task accuracy.

## Executed experiment

`portfolio.agent_eval`: 12 fictional service questions, three repeats, three
policies = 108 actual local-model generations. A shared case/seed controls local
comparisons and policy order rotates. TF-IDF returns up to three documents.
The tool policy has at most three actions and permits search, arithmetic, answer;
arithmetic is an AST numeric DSL with no imports, attribute access or Python eval.
Invalid JSON and exhausted budgets count as failures, not repaired successes.

Strict scoring rejects multiple distinct numeric answers. Confidence intervals
bootstrap questions as clusters, keeping repeated samples together. The small
authored corpus is a smoke test, with shared templates across services, not a
representative production benchmark. Question intervals do not remove that bias.

## Findings and decision

The retrieval component recovered the expected source in every run. End-to-end
exact match was 19.44%; direct and tool-agent policies were 0%. Thus retrieving
evidence is not sufficient: the generation/action stage is the bottleneck. A
production quality gate must reject this model for this task. Median times in
the initial agent run are descriptive: another CPU experiment overlapped, so
they are not clean model-speed comparisons.

`portfolio.adapter_tuning` freezes the base and trains rank-4 query/value adapters
in its final two decoder layers: 15,360 parameters, 48 optimizer steps, 24 training
and four validation examples, 12 disjoint test entities. Masked labels optimize
answer tokens only; validation NLL selects the checkpoint. Test answer NLL fell
6.153 to 0.521, while exact match stayed 33.3%. This is useful negative evidence:
likelihood improvement did not improve answer accuracy. Gold single-document,
greedy adaptation tests are not directly comparable with stochastic top-3 RAG.

## Hosted model arm

`HostedLM` uses the official Responses SDK behind the same generation interface.
It takes an explicit model ID, limits calls/output tokens, captures usage/model
IDs, sets `store=False`, disables hidden SDK retries, and fails on incomplete
responses. Repeats use provider defaults; seeds/temperature are **not** claimed
controlled. Tests use a clearly labeled HTTP fixture, not hosted model evidence.

After configuring an API key privately and choosing a model in your own account:

```sh
python3 -m portfolio.agent_eval --api-model "$PORTFOLIO_MODEL" \
  --max-api-calls 180 --max-output-tokens 1024 --out results/hosted_evaluation
```

The call cap bounds requests, not currency; check provider pricing before running.
No hosted call was made during this build because no credential was configured.
The current small QA suite must be expanded with independent real task cases,
multi-hop reasoning, unanswerable requests and adversarial sources before any
claim that a larger model is suitable for a useful research assistant.

## Reproduction and references

See root commands; saved runs include raw answers, actions, errors and provenance.
Adapters are saved locally in safetensors and excluded from source control.
Base weights are revision-pinned by `portfolio.download_model`.

- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), Apache-2.0.
- [LoRA paper](https://arxiv.org/abs/2106.09685), method reference; this is a small custom implementation.
- [Official Responses reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
