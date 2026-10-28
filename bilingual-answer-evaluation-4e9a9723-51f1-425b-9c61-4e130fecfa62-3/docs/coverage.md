# Coverage by opening

Reviewed against the four linked Microsoft Israel postings on 22 October 2026.
Each role has **two projects**; overlap is intentional. Executed means a local
prototype or experiment ran, not production experience. Partial means the code
addresses a subset, or an external integration is unverified. Personal projects
cannot establish a degree, historical employment, teamwork duration or mastery.

## 200062584 - Data Science: LLMs and agents

[Posting](https://www.microsoftrnd.co.il/JobDetails?JobSeqNo=200062584).
Projects: **Agent Evaluation & Adaptation** + **Reliable Research Service**.

| Requirement area | Evidence | Coverage |
|---|---|---|
| Agents and research automation | Bounded tools; durable retrieval/answer jobs | Executed; typed agent passes the bounded synthetic gate (88/96), broader usefulness unverified |
| LLM experiments, statistical uncertainty | 192 held-out Qwen3-4B outputs plus 108 original baseline generations; clustered bootstrap | Executed; small synthetic suite |
| Failure investigation and iteration | Action traces, invalid-format failures, adapter comparison | Executed; 91.7% vs 43.8% grounded accuracy on bounded synthetic tasks |
| Modeling/evaluation/fine-tuning knowledge | LoRA training; validation-only selection; held-out exact match | Executed on 135M baseline |
| Feasibility and tradeoff communication | Reports, raw outputs, deployment gates | Documented |
| Production coding and repositories | API, persistence, concurrency/fencing tests; Git-ready source | Prototype; not proof of production employment |
| Customer/team collaboration | Acceptance criteria and design notes | Artifact only; prior experience must supply evidence |
| MSc; three semesters; Winter 2028+ graduation | Applicant credentials must establish enrollment and remaining semesters | External eligibility; verify interpretation with employer |
| Preferred PhD | No PhD claimed | Not met by projects |

Hosted model comparison is blocked by missing API credentials; no frontier score
is fabricated. New project work does not replace the production-experience
qualification. Existing Nanose work is described only as previously confirmed.

## 200062523 - Data Science: efficient models

[Posting](https://www.microsoftrnd.co.il/JobDetails?JobSeqNo=200062523).
Projects: **Edge Vision Lab** + **Small-LM Efficiency Lab**.

| Requirement area | Evidence | Coverage |
|---|---|---|
| Vision/model training and experimentation | Teacher/control/distilled student, three seeds | Executed on small digits dataset |
| Compact language models and evaluation | Actual FP32/INT8 pretrained inference | Executed |
| Compression, distillation, quantization | KD loss, reduced architecture, dynamic INT8 | Executed |
| Latency, footprint and resources | Warmups, p50/p95, state size, process RSS | Executed on CPU |
| Power and physical edge constraints | Measurement plan and limits | Unmeasured; needs target device/power access |
| Data pipelines, lifecycle/MLOps | Fixed splits, hashes, seeds, saved histories, CI config | Local pipeline; hosted CI/deployment unverified |
| Reliability, scale and customer impact | Quality gates and documented regressions | Prototype; no customer/scale evidence |
| Python and AI-assisted coding | Runnable Python repository | Executed with AI assistance |
| Cross-functional/silicon collaboration | Tradeoff reports | No teamwork/silicon experience inferred |
| Relevant MSc/PhD | Applicant academic credentials | External credential |

Both optimizations expose negative tradeoffs. No power savings, Windows/NPU
deployment, advanced image/video system or large-scale production claim is made.

## 200047371 - Software Engineering: AI Innovation

[Posting](https://www.microsoftrnd.co.il/JobDetails?JobSeqNo=200047371).
Projects: **Reliable Research Service** + **Agent Evaluation & Adaptation**.

| Requirement area | Evidence | Coverage |
|---|---|---|
| Coding, testing, design and debugging | Python HTTP/SQLite; concurrency, crash and validation tests | Executed locally |
| Agents, tools/APIs, prompts and retrieval | Search/calculator allowlist; local/hosted interfaces | Local executed; hosted contract tests only |
| Quality, relevance, grounding and reliability | Exact-match/evidence recall; source attachment; queue invariants | Partial: claim-level grounding needs human review |
| Logs, traces and monitoring | Durable traces, receipts, aggregate metrics | Implemented; real trace flow exercised |
| Security and responsible AI | Auth, input/body bounds, action DSL, human review | Prototype threat controls; no security certification |
| CI/CD and cloud basics | Workflow definition; operational design | 43 tests passed in GitHub Actions; cloud deployment not performed |
| Requirements, documentation and ownership | Acceptance criteria, runbooks, failure decisions | Artifacts; team interaction not simulated |
| Programming/CS foundations | Algorithms, retrieval, leases, transactions | Project evidence; interview mastery not asserted |
| MSc with three semesters | Applicant academic credentials and remaining semesters | External eligibility |
| Preferred PhD and experience durations | No new historical duration inferred | Confirm from existing experience; projects cannot backdate it |

The local service works. Calling it production-ready or sovereign-cloud deployed
would exceed the evidence. A stronger model and broader eval are a release gate.

## 200028527 - Security Research

[Posting](https://www.microsoftrnd.co.il/JobDetails?JobSeqNo=200028527).
Projects: **Security Telemetry Lab** + **Evidence-Based Triage Service**.

| Requirement area | Evidence | Coverage |
|---|---|---|
| Detection, hunting and telemetry analysis | Temporal rules vs anomaly baseline; held-out evaluation | Executed on synthetic events |
| KQL and large telemetry | Two reference KQL translations; bounded schema | KQL not executed; scale not demonstrated |
| Automated/AI-assisted investigation | Security retrieval/rule evidence through durable API | Actual local-model flow; review required |
| Coverage refinement and root causes | Slow-attack misses; maintenance false positives; gap table | Executed analysis; synthetic scenarios |
| Architecture/threat-model feedback | Source contracts, threat model, engineering recommendations | Documented |
| Real attacks/vulnerabilities/nation-state work | No real incident data or attribution | Not demonstrated |
| OS/network/distributed fundamentals | Local HTTP, processes, leases, transaction recovery | Partial; no kernel/Windows-internals evidence |
| Cloud/Kubernetes/containers | Synthetic identity/workload event semantics | Concepts only; no cluster deployment |
| Python tooling, initiative and communication | Detection/triage code and investigation report | Project artifacts; no team delivery inferred |
| Degree with one year remaining | Applicant academic credentials and graduation date | External eligibility |

This role retains the largest gaps. The projects are defensive research exercises,
not a claim of operational incident response or real cloud-security expertise.

## Concrete remaining gates

1. Configure an authorized model API account; run the hosted arm and extend the
   task suite beyond templated numeric QA before claiming a useful agent product.
2. Hosted CI passed (43 tests, run 39810558301). A public cloud deployment and production load testing remain separate work.
3. Execute KQL and native event ingestion in an authorized cloud lab; access target
   edge hardware and a power meter for power claims. These cannot be invented.
4. Run, modify and explain the code personally before using it to evidence skills
   in an interview. AI assistance is recorded in README and project CV headings.

No applications were submitted. Applied Sciences is outside this four-role set.
