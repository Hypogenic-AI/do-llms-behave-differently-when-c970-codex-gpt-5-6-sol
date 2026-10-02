# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-02T12:23:46.626939Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/do-llms-behave-differently-when-c970-codex-gpt-5-6-sol`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
Phase complete (2026-10-02). Gathered and validated 17 PDFs/157 chunks, 4 local dataset packages or subsets, and 11 code repositories. Deliverables: `literature_review.md`, `resources.md`, `planning.md`, `papers/README.md`, `datasets/README.md`, and `code/README.md`; completion marker exists at `.resource_finder_complete`.

Direction budget is frozen to: (1) paired black-box human-style vs LLM-style behavioral audit, (2) factorial decomposition of source style from length/structure/clarity/politeness/source label, and (3) white-box authorship-representation mediation/intervention. Scores and rejected directions are in `planning.md`.

Key evidence: source/context recognition is robust, but downstream effects are model/task dependent; probe decodability, verbalization, and behavior often dissociate. Format and generator identity are major confounds. Use JBB for refusal/compliance, Anthropic evals for sycophancy, WildChat for natural prompt seeds, and EvalDetectBench replication tables for generator/probe calibration controls.

Next phase: construct semantically matched prompt variants; blind-check equivalence and perceived authorship; pre-register paired behavioral outcomes and hierarchical analysis; run the surface-factor experiment before training held-out-generator authorship probes; use steering only as a final causal stress test.

Uncertainty: WildChat source card has no declared license locally; full EvalDetectBench logs were not downloaded (1.9 GB), only the complete compact paper-replication subset. White-box code was inspected but not GPU-executed. Paper-finder relevance scores were noisy; all 74 cached records were title/abstract screened manually.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
Experiment runner complete (2026-10-02). Executed all three frozen directions with 120 fixed items, 625 unique scored API-model responses, a source-label subset, a 105-pair Qwen3-8B authorship probe, and 128 fixed-text steering generations with five random controls. Main findings: perceived-AI manipulation +13.1 to +41.9 points by task; none of six preregistered behavioral contrasts survived Holm correction; probe test AUROC 0.907 versus 0.601 surface baseline; authorship/evaluation cosine 0.242; ± authorship steering produced the same +6.25-point aggregate change, matched by a random direction, so the causal criterion failed. Outputs: `REPORT.md`, `README.md`, `planning.md`, `src/`, `results/`, and `figures/`. Repeated analysis hashes matched; `python -m compileall -q src` and `src/validate.py` passed (summary: `results/evaluations/validation_summary.json`). OpenRouter usage: 1,879 unique successful calls, 423,406 prompt and 277,499 completion tokens, reported cost $2.13. Remaining uncertainty: pilot-sized intervals, LLM-generated “human-style” text, judge dependence, harmful refusal/sycophancy ceiling effects, and one open model/steering layer.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
