# Prompt Authorship Style and LLM Behavior

This project tests whether semantically matched prompts that read human-written versus LLM-written change safety-relevant model behavior. It combines current API-model experiments, surface/paraphrase/source-label controls, a held-out Qwen3-8B authorship probe, and activation steering with norm-matched random controls.

## Key findings

- The manipulation was recognized: LLM-style text gained 13–42 points in judged AI-authorship probability across tasks.
- None of six preregistered behavioral contrasts across harmful compliance, sycophancy, and accuracy was significant after Holm correction.
- Qwen3-8B decoded style strongly (held-out AUROC 0.907) versus a surface-only baseline (0.601).
- The authorship direction only modestly overlapped evaluation awareness (cosine 0.242).
- ± authorship steering shifted the probe logit by ±17.08 but did not beat five random directions behaviorally; decodability did not imply causal use.

See [REPORT.md](REPORT.md) for methods, tables, figures, limitations, and interpretation.

## Reproduce

Prerequisites: `uv`, an NVIDIA GPU for the white-box arm, `OPENROUTER_KEY`, and optionally `HF_TOKEN`.

```bash
source .venv/bin/activate
uv sync

# Fixed data sample and audit
python src/prepare_data.py

# API experiment (existing JSONL caches make this resumable)
python src/run_behavioral.py

# Main statistics and figures
python src/analyze.py

# White-box probe and gated steering; the project-local compiler shim is
# needed by this PyTorch/Triton build in compiler-free containers.
CC="$PWD/tools/cc" python src/run_probe.py
python src/analyze_steering.py
```

The existing raw caches reproduce all analysis without new API calls. Model APIs are time-varying; exact IDs, access date, returned metadata, usage, cost, and response hashes are retained in the artifacts.

## Structure

```text
planning.md                 preregistered design and direction budget
literature_review.md        prior-work synthesis
src/prepare_data.py         deterministic stratified sample
src/run_behavioral.py       rewrites, validation, responses, label ablation
src/run_probe.py            grouped probe, direction comparison, steering
src/analyze*.py             statistics and figures
results/data/               fixed inputs and data audit
results/model_outputs/      raw generated responses
results/evaluations/        tidy data, scores, tests, probe artifacts
figures/                    report figures
REPORT.md                   complete research report
```

Python dependencies are pinned by `uv.lock`; the random seed is `20261002`.
