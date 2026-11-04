# Applied AI Portfolio

Numeric service-policy questions require current evidence, correct arithmetic and a defensible decision to abstain. Retrieval alone does not ensure a correct answer, and valid calculations can still follow an incorrectly interpreted question.

## What it does

Applied AI Portfolio is a local Python research suite for traceable service-policy question answering, LLM adaptation, model efficiency and synthetic security telemetry. Its studies document higher grounded correctness for a typed agent than text RAG on the tested questions, unchanged exact match after LoRA, and mixed quantization tradeoffs.

Compare local agents and adapter tuning on held-out data, evaluate answer correctness and evidence support separately, and measure model-efficiency and telemetry-detection tradeoffs through reproducible experiments.

## Solution and technologies

Numeric QA compares a TF-IDF text-RAG baseline with a typed Qwen3-4B agent. The agent separates schema-constrained planning from deterministic execution: versioned fact lookup, unit and arithmetic checks, source provenance and abstention. The evaluation runner separates development/test entities and records outputs and paired uncertainty.

A separate PyTorch LoRA pipeline freezes base weights and selects checkpoints by validation answer-token loss. An HTTP API and SQLite queue support research jobs and inspectable traces. Independent modules evaluate vision distillation and dynamic INT8 quantization, small-language-model CPU performance, and temporal rules versus Isolation Forest on synthetic telemetry.

Python · PyTorch · NumPy · scikit-learn: TF-IDF and Isolation Forest · Qwen3-4B: GGUF CPU inference · SmolLM2-135M-Instruct · Custom LoRA adapters · safetensors · SQLite · JSON Schema · Python AST

## Quick start

Python 3.11+; CPU supported. Install PyTorch's CPU wheel separately if desired.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m portfolio.edge_vision --out results/edge_vision
python -m portfolio.telemetry --out results/telemetry
python -m portfolio.download_model
python -m portfolio.agent_eval --model-path /path/to/SmolLM2-135M-Instruct --out results/agent_eval
python -m portfolio.small_lm --model-path /path/to/SmolLM2-135M-Instruct --out results/small_lm
python -m portfolio.adapter_tuning --model-path models/SmolLM2-135M-Instruct
python -m portfolio.acceptance --model-path models/SmolLM2-135M-Instruct
```

The first two experiments work offline. LLM experiments require actual local
model weights; they **fail** when weights are absent rather than fabricate scores.
Never submit personal CVs, credentials or private company data to this repository.

## Results

On 48 authored synthetic questions across six held-out service entities, repeated twice per system, the typed agent achieved 88/96 (91.7%) answers correct with expected evidence versus 42/96 (43.8%) for text RAG. Both used Qwen3-4B-Instruct-2507-Q4_K_M.gguf on CPU. Correct abstentions were 36/36 versus 18/36; neither system produced invalid or failed outputs. Subtraction and service-name interpretation errors remained.

A separate single-seed PyTorch LoRA study reduced held-out answer-token NLL from 6.153 to 0.521 while exact match remained 33.3% on 12 authored test cases.

In the separate SmolLM2-135M-Instruct QA evaluation, TF-IDF retrieval achieved 19.44% strict numeric exact match; direct and tool-agent policies scored 0% on 12 authored synthetic questions repeated three times.

On a fixed 360-image digits test set shared across three training seeds, the distilled FP32 student averaged 97.59% accuracy versus 97.69% for its dynamic INT8 variant. Serialized state fell from 9.3 to 5.4 KiB, while median batch-1 CPU latency increased from 0.0100 to 0.0312 ms.

For SmolLM2-135M-Instruct on CPU, dynamic INT8 reduced serialized state from 513.2 to 236.6 MiB and increased median generation speed from 14.88 to 47.23 tokens/s. Resident memory increased from 960.7 to 1540.6 MiB, and perplexity on six authored passages worsened from 120.26 to 258.58.

On 240 synthetic held-out telemetry episodes, temporal rules recorded F1 of 0.746 versus 0.727 for Isolation Forest. Recall was 73.33% versus 93.33%, with false-positive rates of 7.78% versus 21.11%, respectively.

## Scope and limitations

QA and telemetry use authored synthetic cases with shared task families; the vision study uses a small educational dataset.

The typed agent queries structured records while RAG receives six retrieved text records; differing evidence access limits causal attribution.

Local CPU measurements do not establish production capacity, live-cloud deployment, physical edge-device performance or power consumption.

## Documentation

[Technical guide and current validation](DOCUMENTATION.md) · [Code map](CODE_MAP.md) · [Walkthrough](EXERCISES.md)

[Historical and supporting records](HISTORY.md) retain earlier reports and validation records. Earlier published versions remain available through Git history.

[Role-specific project mappings](EMPLOYER_CONTEXT.md) are documented separately.
