# Resources catalog

## Summary

The workspace contains 17 validated papers, four dataset packages/subsets, 11 cloned repositories, full-text chunk manifests, small documentation samples, and a project-local Python environment. The selected resources support a three-part experiment: behavioral comparison, factorial surface controls, and latent-representation mediation.

## Papers

| Count | Location | Notes |
|---:|---|---|
| 17 PDFs | `papers/*.pdf` | All readable; 11 user-specified papers plus 6 directly relevant discovered papers. |
| 157 PDF chunks | `papers/pages/` | Seventeen manifests map chunks to source-page ranges. |

See `papers/README.md` and `literature_review.md` for bibliographic details and synthesis.

## Datasets

| Name | Source | Local size/content | Task | Location |
|---|---|---|---|---|
| WildChat-1M sample | Hugging Face `allenai/WildChat-1M` | 5,000 rows; ~19 MiB | Natural human prompt seeds and deployment controls | `datasets/wildchat_5k/` |
| JBB-Behaviors | Hugging Face `JailbreakBench/JBB-Behaviors` | 100 harmful, 100 benign, 300 judge-comparison rows | Refusal and harmful compliance | `datasets/jbb_behaviors/` |
| Anthropic model-written evals | Hugging Face `Anthropic/model-written-evals` | 30,168 sycophancy plus persona/risk/WinoGenerated data | Sycophancy and model-written controls | `datasets/anthropic_model_written_evals/` |
| EvalDetectBench replication subset | Hugging Face `el7982/aware-bench` | 11 CSV tables; ~2.1 MiB | Generator confounds and probe calibration | `datasets/evaldetectbench_replication/` |

Data files are local but ignored by Git. Reproduction commands, schemas, licenses, and sample paths are in `datasets/README.md`.

## Code repositories

| Name | Purpose | Location |
|---|---|---|
| ai-ai-bias | Paired human/LLM presentation and preference pipeline | `code/ai-ai-bias/` |
| eval-awareness | Black-box evaluation/deployment classification | `code/eval-awareness/` |
| evaluation-awareness-probing | Minimal probe/SAE experiments | `code/evaluation-awareness-probing/` |
| test-awareness-steering | Probe training, steering, and action compliance | `code/test-awareness-steering/` |
| RepreGuard | Representation-based machine-text detector | `code/repreguard/` |
| mgt-probes | Robust layer-wise and concatenated authorship probes | `code/mgt-probes/` |
| eval-awareness-format | Controlled format/context 2×2 and rewrite scripts | `code/eval-awareness-format/` |
| decomposing-eval-awareness | EvalAwareBench and recognition/propensity analysis | `code/decomposing-eval-awareness/` |
| evaluation-awareness-representation | Probe AUROC, correlation/MI, and controls | `code/evaluation-awareness-representation/` |
| interlocutor-awareness | Model-family recognition and identity-conditioned behavior | `code/interlocutor-awareness/` |
| evaldetectbench | Prompt calibration and generator-confound correction | `code/evaldetectbench/` |

Exact remotes, commits, requirements, and entry points are in `code/README.md`.

## Search and selection notes

The paper-finder service timed out interactively but later produced a cached 74-record file. Every title/abstract record was screened; three new papers met the criteria, three duplicated already staged papers, and the remainder were generic evaluation, unrelated applications, or background methods. The fallback combined direct arXiv pages, Semantic Scholar links, paper code/data pages, and backward/forward citation chaining. Selection favored empirical studies with matched interventions, explicit confound controls, public data/code, or direct relevance to internal representations and downstream behavior.

The eleven papers and three datasets named in the research specification were handled first. Six additional papers were added because they materially change experimental design: format confounding, limited average behavioral effects, factor decomposition, interlocutor identity, scaling, and generator-identity measurement bias.

## Challenges and workarounds

- The paper-finder endpoint did not return; all searches and screening were logged manually.
- Full WildChat-1M is large and contains privacy-sensitive metadata. A reproducible 5,000-row local sample was validated, and only sanitized examples are documentation-visible.
- Full EvalDetectBench artifacts total about 1.9 GB, mostly raw model logs. The complete compact paper-replication subset was downloaded instead.
- Several white-box pipelines require large model weights and GPUs. Repositories were validated by documentation and entry-point inspection; no expensive model run was claimed.
- The `decomposing-eval-awareness` repository initially failed during a full shallow clone. It was successfully cloned with blob filtering and a sparse checkout at the published commit. Two incomplete temporary transfers were moved outside the workspace to `/tmp` and are recoverable only until the container is recycled.

## Experiment recommendations

1. **Primary datasets:** JBB harmful/benign for refusal/compliance; Anthropic sycophancy for agreeableness; WildChat as natural seeds and a deployment-style control. Add a small verifiable QA benchmark during experiment setup for accuracy.
2. **Behavioral baseline:** Adapt `ai-ai-bias` into randomized within-item comparisons and direct-response scoring.
3. **Confound baseline:** Reuse the 2×2 logic and length matching from `eval-awareness-format`; incorporate EvalAwareBench’s independent factor toggles.
4. **Mechanistic baseline:** Use `mgt-probes` with held-out generators/domains, surface-feature baselines, and random directions. Use the correlation/MI controls from `evaluation-awareness-representation` and generator harmonization from `evaldetectbench`.
5. **Metrics:** Paired refusal/compliance differences, harmfulness, sycophancy, accuracy, AUROC/calibration for source recognition, and hierarchical model×task interactions with bootstrap confidence intervals.
6. **Decision rule:** The central hypothesis requires a behavioral source-style effect. Probe decodability alone is insufficient; evidence for an internal mechanism additionally requires mediation that survives surface controls and a controlled intervention that shifts behavior.

The Phase 1 direction budget and rejected alternatives are recorded in `planning.md`.

## Experiment execution artifacts (2026-10-02)

The experiment runner used the staged JBB, Anthropic sycophancy, MMLU, and evaluation-awareness resources to execute all three retained directions. Real API outputs are in `results/model_outputs/`; scored/tidy results, exact tests, clustered surface adjustments, the Qwen3-8B probe, direction-cosine analysis, steering controls, cost accounting, and hashes are in `results/evaluations/`. Code is in `src/`, figures in `figures/`, the preregistration in `planning.md`, and the final synthesis in `REPORT.md`. The run used an RTX A6000 and exact Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
