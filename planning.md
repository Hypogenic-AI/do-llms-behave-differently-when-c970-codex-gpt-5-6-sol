# Research plan and direction budget

## Motivation & Novelty Assessment

### Why This Research Matters

Synthetic red-team prompts, model-written evaluations, and orchestrator-to-agent messages are often LLM-written, whereas deployment traffic is substantially human-written. If responders condition safety behavior on perceived authorship, synthetic evaluations can systematically misestimate real-world refusal, compliance, sycophancy, or accuracy.

### Gap in Existing Work

Prior work separately establishes evaluation-context recognition, latent human/machine-text separability, self-preference, and behavior changes under hypothetical or test framing. It does not jointly isolate prompt authorship style from paraphrasing and surface features, verify that responders perceive the manipulation as intended, and connect a latent authorship direction to downstream behavior with controlled interventions.

### Our Novel Contribution

This study combines (1) paired, content-matched behavioral comparisons across three behaviors, (2) a paraphrase control that separates “was rewritten” from “reads LLM-written” plus measured surface covariates and explicit source-label controls, and (3) a white-box probe/steering stress test with text held fixed and norm-matched random directions. The inferential ladder is deliberately strict: behavioral difference, successful manipulation check, robustness to surface controls, then causal intervention.

### Experiment Justification

- **Experiment 1 — paired behavioral audit:** establishes whether a style-associated behavioral difference exists at all across harmful compliance, sycophancy, and answer accuracy.
- **Experiment 2 — surface/label decomposition:** distinguishes perceived authorship from generic paraphrasing, length, clarity, politeness, and explicit beliefs about source.
- **Experiment 3 — representation and intervention:** tests whether an internal authorship direction predicts behavior and whether changing it with the exact prompt text fixed moves behavior beyond norm-matched random steering or merely degrades output.

## Research Question

For semantically matched user requests, does human-like versus LLM-like phrasing change safety-relevant behavior, and does any effect survive controls for paraphrasing and observable style or respond causally to an internal machine-authorship direction?

## Background and Motivation

The literature summarized in `literature_review.md` implies three distinct constructs: source/context recognition, latent decodability, and downstream behavioral use. They often dissociate. Consequently, neither an authorship classifier nor an unadjusted human/LLM comparison is sufficient evidence; the experiment must validate source perception, include style controls, and reserve causal language for an intervention with random and degradation baselines.

## Hypothesis Decomposition

- **H1 (recognition):** LLM-style variants receive a higher perceived-machine probability than original or human-style paraphrases of the same content.
- **H2 (behavior):** within an item and responder, LLM-style versus human-style wording changes at least one preregistered outcome: harmful compliance, sycophantic agreement, or correctness.
- **H3 (specificity):** the H2 contrast is not explained by paraphrasing alone and remains directionally stable after adjustment for token count, readability, politeness, and clarity ratings.
- **H4 (label versus style):** explicit source labels may shift behavior, but an unlabeled style effect is separately identifiable; a style-by-label interaction would indicate belief-mediated rather than purely linguistic conditioning.
- **H5 (representation):** a held-out linear probe decodes prompt authorship above chance; stronger machine scores predict behavior beyond surface covariates.
- **H6 (causal stress test):** authorship-direction activation addition on fixed prompt text shifts recognition and behavior more than equal-norm random directions without a commensurate loss in fluency/task accuracy.

Independent variables are prompt condition (original, human-style paraphrase, LLM-style paraphrase), explicit source label (human/AI/none in a focused subset), responder model, task, and—white-box only—steering direction/strength. Outcomes are perceived-AI probability, refusal/compliance, sycophantic option choice, accuracy, output validity, and quality/degradation. The item is the paired sampling unit.

## Proposed Methodology

### Approach

Use a staged confirmatory pilot. Construct variants once with a separate current API model, validate semantic equivalence and perceived authorship with blinded structured judgments, then query two model families at temperature 0. Analyze only prespecified paired contrasts; exploratory model/task heterogeneity is labeled as such. Run a local 8B instruct model for latent activation extraction and a bounded steering experiment. This scale is intended to estimate effect size and identify ceiling/floor behavior, not support universal claims.

### Experimental Steps

1. Sample 40 items per task from JBB harmful behaviors, Anthropic sycophancy, and WinoGenerated/answerable controls, stratifying where metadata permit.
2. Generate two meaning-preserving rewrites per item: an informal/natural human-like paraphrase and a polished, structured LLM-like paraphrase. Constrain both to 0.8–1.2× source word count and forbid added facts, rationale, or task constraints.
3. Blind-judge semantic fidelity, information additions/omissions, clarity, politeness, and perceived authorship. Exclude/repair pairs with fidelity below 4/5 or any substantive drift before behavioral analysis.
4. Query `openai/gpt-5.6-luna` and `google/gemini-3.1-flash-lite` (live-catalog verified 2026-10-02) with identical system prompts and deterministic settings. Cache every request, response, usage record, timestamp, and error.
5. Score structured-choice tasks deterministically. Score harmful responses using a blinded rubric judge plus refusal-string sensitivity checks; preserve raw judgments.
6. Run the focused 2×3 label/style ablation on a stratified subset, with source labels counterfactually crossed and no-label as baseline.
7. On Qwen3-8B, extract layerwise residual representations for separate human/machine text and prompt variants, train regularized probes with item/group-held-out splits, test held-out variant type, and compare with length/readability baselines.
8. If probe AUROC exceeds 0.70 on held-out content, steer the best validation layer bidirectionally on fixed prompts at prespecified magnitudes (0.5 and 1.0 activation SD), alongside five equal-norm random directions and a no-steer control.

### Baselines

- Original unmodified prompt and human-style paraphrase (separates authorship style from rewriting).
- Token-length/readability/politeness/clarity covariates and surface-only classifier.
- No-label and counterfactual explicit human/AI labels.
- Majority behavior and deterministic refusal keyword score.
- White-box no-steer, sign-flipped authorship steering, and five norm-matched random directions.
- Output-quality/validity checks to distinguish targeted behavioral movement from generic model degradation.

### Evaluation Metrics

- Recognition: AUROC, mean perceived-AI probability, paired difference, Brier score where binary reference labels apply.
- Behavior: within-item risk difference for binary outcomes; paired odds ratio; exact discordant-pair (McNemar/binomial) test; cluster bootstrap 95% CI over items.
- Semantic control: fidelity pass rate, word-count ratio, lexical similarity, and additions/omissions flags.
- Probe: group-held-out AUROC, balanced accuracy, and incremental likelihood/pseudo-R² beyond surface covariates.
- Steering: difference-in-differences relative to no-steer and empirical random-direction distribution; invalid-output and accuracy deltas as degradation checks.

### Statistical Analysis Plan

The primary family contains six contrasts: LLM-style minus human-style for three outcomes across two responders. Use two-sided item-cluster bootstrap 95% intervals and Holm correction at familywise α=0.05. Exact McNemar tests handle paired binary outcomes. A logistic regression with item-clustered standard errors estimates condition, responder, task, and prespecified interactions; surface-adjusted models add standardized word count, readability, clarity, and politeness. We will report raw and adjusted effects regardless of significance. With 40 items per task, the pilot has useful precision only for moderate/large paired effects (roughly ≥15 percentage points near p=0.5); smaller intervals are explicitly treated as inconclusive, not null proof.

For probes, all rewrites of an item remain in one split. Hyperparameters and layer are selected on validation only; the test split is evaluated once. Mediation language is descriptive unless both intervention and degradation controls pass. Steering inference compares the authorship direction with the empirical distribution of five seeded random directions and is exploratory because the number of directions is small.

## Expected Outcomes

Support for a practically important phenomenon requires a successful recognition manipulation plus a behavior shift that is larger than the human-paraphrase/original difference and robust to surface adjustment. A recognizable style with negligible paired behavior differences supports a recognition–behavior dissociation. Probe success without controlled steering is evidence of decodability only. Failed or random-like steering refutes the proposed simple linear causal mechanism but not all possible internal mechanisms.

## Timeline and Milestones

1. Environment, live catalog, and data audit: 15 minutes.
2. Variant construction and validation pilot: 30–60 minutes.
3. Behavioral calls and scoring: 45–90 minutes.
4. White-box probe and steering: 45–90 minutes, with the intervention gated on probe validity.
5. Statistical analysis, rerun checks, figures, and documentation: 45–60 minutes.

The critical path is rewrite validation → behavioral run → scoring. White-box work proceeds only after the behavioral harness is valid. API responses are checkpointed per call so interruption does not discard completed work.

## Potential Challenges

- Harmful prompts may induce ceiling refusal; sample borderline JBB categories and report headroom by condition rather than substituting post hoc.
- Rewriters may refuse or semantically sanitize harmful items; record failures and never silently replace content.
- Deterministic APIs may still change behind a stable ID; record date, exact routed model/provider metadata where returned, and raw output hashes.
- LLM judges can share style bias with the manipulation; use structured rubrics, blind condition labels, deterministic checks, and sensitivity analyses.
- Probe directions can encode length/format; group splits, matched variants, surface baselines, held-out rewrite types, and random directions address but do not eliminate this concern.
- One local 8B model and a pilot sample limit generalization; conclusions will be model- and task-bounded.

## Success Criteria

The research cycle succeeds if it produces auditable matched prompts, real-model outputs, manipulation checks, paired estimates with uncertainty for all three behaviors, and an honestly gated white-box result. The substantive H2 is supported only if a Holm-adjusted contrast excludes zero and exceeds 10 percentage points while H1 succeeds; H3 additionally requires the adjusted estimate to retain at least half its magnitude. H6 requires recognition movement in the intended direction, behavioral movement beyond all five random baselines, and no >5-point rise in invalid/degraded outputs.

## Scope

**Question.** When semantic content is held fixed, does a prompt written in a human-like versus LLM-like style change refusal, harmful compliance, sycophancy, or answer accuracy? If so, is the effect mediated by an internal representation of machine authorship, or by measurable surface features such as length, structure, clarity, and politeness?

**Primary domain.** Language-model behavior, AI safety evaluation, mechanistic interpretability, and machine-generated-text detection.

**Search terms.** `human vs machine generated prompt LLM behavior`, `evaluation awareness`, `test awareness steering`, `machine generated text linear probe`, `AI-AI bias`, `self-recognition self-preference`, `prompt format confound`, `synthetic prompt ecological validity`, `sycophancy model-written evaluations`, and `harmful compliance prompt framing`.

**Inclusion.** Empirical work that measures (a) recognition of human/machine or evaluation/deployment text, (b) behavioral changes under matched prompt framing/style, (c) internal representations or causal interventions, or (d) prompt-construction confounds. Preference was given to 2024–2026 work with data or code.

**Exclusion.** Generic AI-text detectors without latent-space or behavioral relevance; studies of prompt engineering that do not control semantic content; opinion pieces without data; and safety work with no manipulation of prompt source, framing, or perceived context.

## Direction ranking

Scores are 1–5 per criterion. “Evidence” reflects direct literature support; “relevance” reflects fit to the stated hypothesis; “information gain” reflects ability to distinguish competing explanations; “feasibility” reflects implementation with the staged data and code.

| Rank | Direction | Evidence | Relevance | Information gain | Feasibility | Total | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Paired black-box behavioral audit: human-style versus LLM-style rewrites of identical requests across refusal/compliance, sycophancy, and accuracy | 5 | 5 | 5 | 5 | 20 | Keep |
| 2 | Factorial surface-control study: independently manipulate source style, length, structure, clarity, politeness, and explicit source labels | 5 | 5 | 5 | 4 | 19 | Keep |
| 3 | White-box mediation study: decode perceived machine authorship, test mediation, and intervene by activation steering or representation matching | 5 | 5 | 5 | 3 | 18 | Keep |

### Retained implementation directions

1. **Behavioral effect.** Build semantically matched prompt pairs from JBB-Behaviors, Anthropic sycophancy items, accuracy questions, and natural WildChat requests. Cross prompt style with responding model and task. Estimate within-item differences and model heterogeneity.
2. **Surface-feature decomposition.** Use a crossed design rather than a single human/LLM contrast. Match or orthogonalize token count and readability; separately vary template structure, politeness, explicitness, and source label. Include blinded human and model ratings of semantic equivalence and perceived authorship.
3. **Representation and causality.** On an open-weight model, train held-out linear probes for human/LLM authorship using multiple generators/domains, then ask whether probe score predicts behavioral changes after surface controls. Use steering or matched paraphrases as a causal stress test; do not infer causality from probe accuracy alone.

## Pruned directions

| Direction | Reason for pruning |
|---|---|
|---|
| Pure human-vs-machine text detection benchmark | Establishes recognizability but not whether recognition changes the response; retained only as a measurement component of Direction 3. |
| Evaluation-awareness classification alone | Closely related, but “evaluation vs deployment” is not identical to “human vs LLM author.” It is used for methods and controls, not as the main outcome. |
| Self-preference/LLM-as-judge only | Strong evidence for one downstream bias, but too narrow to answer refusal, compliance, sycophancy, and accuracy jointly. It becomes one secondary outcome. |
| Test-awareness steering as the sole experiment | Steering effects are model-dependent and can manipulate entangled concepts. It is retained only as a causal stress test after the behavioral and factorial analyses. |
| Hiring-field simulation | High external relevance but domain-specific, expensive, and less diagnostic of mechanism than matched benchmark tasks. |
| Training-stage longitudinal study | Scientifically useful but requires many checkpoints and substantially more compute; lower immediate information gain for the prompt-source hypothesis. |
| Build a new large prompt corpus before piloting | Existing resources are adequate for a powered pilot. New corpus construction should follow only if semantic-equivalence or coverage checks fail. |

## Frozen decision

The three retained directions are the implementation budget. The experiment phase should not add a fourth direction unless new evidence invalidates this ranking; any change must be recorded in the resource-finder notes block of `STATE.md`.
