# Security Telemetry and Evidence-Based Triage

The detection lab and the security mode of the durable research service share
transport and evidence contracts. The structured alert extension adds bounded
investigation and replayable experiments; see the [root guide](../README.md).

## Detection lab

The synthetic generator includes login bursts, privilege changes followed by
secret access, workload execution, slow guessing, normal retries and authorized
production maintenance. IPs use documentation ranges. No real customer or attack
logs are included. Actor IDs, random seeds and time ranges are disjoint across
240 training, 120 validation and 240 test episodes; scenario templates are shared.

Strict schemas reject conflicting IDs; exact duplicates are deduplicated.
Rules correlate the same actor/source or resource within 120 seconds. A rule
threshold is selected using validation F1, with conservative tie-breaking.
Isolation Forest fits only benign training episodes and selects its cutoff on
validation. Labels never enter features. Test data is used only for evaluation.

On the held-out synthetic set: rules precision 75.86%, recall 73.33%, F1 0.746,
false-positive rate 7.78%. Isolation Forest recall rises to 93.33%, while its
false-positive rate rises to 21.11%. These are scenario-generator scores, not
claims of protection against real attacks or massive datasets.

## Threat model and root-cause analysis

| Observed gap | Cause | Engineering change to investigate |
|---|---|---|
| Slow guessing missed | Events lie outside a short window | Separate longer-window detector; validate benign retry rate before enabling |
| Authorized maintenance flagged | Event sequence alone lacks intent | Join with change approvals, workload identity and privilege baseline |
| More anomaly false positives | Event volume is unusual but legitimate | Calibrate per actor/workload and retain a human review queue |
| Missing audit source | A detector cannot infer unrecorded events | Coverage contract for actor, source, operation, resource, timestamp and outcome |
| Out-of-order/duplicate delivery | Multiple producers/retries | Normalize ordering/deduplicate IDs; define event-time watermark for streaming |

`queries/detections.kql` translates the two temporal rules for a documented
PortfolioEvents schema. It has **not** run on Kusto/Azure. Dedup assumes upstream
conflicting IDs were already rejected. A production pipeline needs an explicit
mapping from native Kubernetes/audit/identity logs and representative throughput
tests; the synthetic schema is not a claim of native cloud ingestion.

## Triage service

Security mode retrieves a bounded evidence set, adds matching rule records, calls
the actual local model and saves the response, evidence IDs and trace. The real
HTTP acceptance run verifies this path. The prompt treats retrieved text as data,
but this is not a security proof; generated findings require review. No blocking,
remote commands, exploitation or automatic incident attribution is implemented.

See [operations](operations.md) for authentication, fencing, retries, threat model
and limitations. OS/network fundamentals are represented by local HTTP/process/
SQLite behavior; Windows internals, real Kubernetes operations and nation-state
investigations have not been evaluated by this lab.

[Official KQL join reference](https://learn.microsoft.com/en-us/kusto/query/join-operator?view=microsoft-fabric).
