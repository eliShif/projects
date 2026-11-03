# Small Language Model Efficiency Lab

The efficient-model role explicitly involves compact language models. This
project studies a quality/resource tradeoff, separately from the agent product's
need for a model that can reliably complete tasks.

SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, runs
with actual pretrained weights. FP32 and dynamic INT8 Linear-module variants run
in separate child processes. The final measurements were repeated with no other
portfolio model experiment running. Embeddings stay FP32; this is not full INT8.

Both variants use the same six authored prose passages, prompt lengths, tokenizer,
greedy generation, warmups and one CPU thread. Perplexity is token-weighted over
the passages. These passages are a smoke test, not WikiText or a standard benchmark.
Raw timing samples, generated outputs, serialized state bytes and Linux RSS after
loading are recorded in each variant's JSON.

The measured serialized state shrank from 513.2 to 236.6 MiB, median prefill from
145.49 to 39.73 ms, and generation grew from 14.88 to 47.23 tokens/s. But perplexity
worsened from 120.26 to 258.58; process RSS rose from 960.7 to 1540.6 MiB. The
conversion can retain allocator memory, so a smaller saved state is not evidence
of lower runtime RAM. RSS is after load, not measured peak memory.

Decision: current INT8 configuration fails a quality-preserving optimization
goal. Investigate selective quantization and target-device backends with a
representative evaluation set before shipping. No energy, physical edge-device,
Windows or production-scale deployment claim is supported.

Run `python -m portfolio.small_lm --model-path models/SmolLM2-135M-Instruct
--out results/small_lm` on one shell line. Results vary by CPU/allocator/version;
do not compare milliseconds across unrelated machines as a model-only effect.

[Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) and
[PyTorch quantization API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html).
