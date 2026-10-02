# Working title

**Recognized but Not Used: A Paired Audit of Human-Like and LLM-Like Prompt Style**

## Abstract sketch

- Context: Synthetic evaluations and agent-to-agent traffic often use model-written prompts; transfer is questionable if responders condition behavior on perceived authorship.
- Gap: Prior work studies evaluation awareness, source detection, or self-preference separately, but does not combine content-matched prompt-style contrasts, multiple behaviors, surface controls, and a causal representation test.
- Approach: Audit 120 items across harmful compliance, benign over-refusal, sycophancy, and accuracy with paired rewrites, two API responders, blinded judgments, surface/paraphrase/label controls, and Qwen3-8B probing and steering.
- Key results: judged AI authorship rises 13.1--41.9 points, but none of six primary contrasts is Holm-significant; Qwen authorship AUROC is 0.907 versus 0.601 for surface features; both steering signs move behavior by +6.25 points and one random direction matches this shift.
- Significance: The result is a bounded null and recognition--behavior dissociation, not an equivalence proof.

## 1. Introduction

- Hook: evaluations increasingly rely on synthetic prompts, while deployment includes human prompts.
- Importance: a responder that detects synthetic style could distort measured refusal, compliance, sycophancy, or accuracy.
- Gap: recognition, behavioral use, and causal control are distinct and rarely tested together.
- Approach: paired content-preserving rewrites; manipulation check; behavioral audit; surface/paraphrase/label controls; held-out probe and random-controlled steering.
- Quantitative preview: +13.1 to +41.9 point recognition shift; 0/6 corrected primary effects; 0.907 probe AUROC; failed steering criterion.
- Contributions:
  1. Formulate an inferential ladder separating recognition, behavioral association, and causal use.
  2. Conduct a paired four-task, two-responder pilot with preregistered multiplicity control.
  3. Decompose paraphrase, surface, and explicit-label alternatives.
  4. Test a latent authorship direction with held-out evaluation and norm-matched controls.
- Evidence: `manipulation_checks.csv`, `paired_primary.csv`, `adjusted_models.json`, `probe_results.json`.
- Citations: Needham; Abdelnabi and Salem; Nguyen et al.; Panickssery et al.; Chen et al.

## 2. Related work

### Recognition and source preference

- Panickssery et al., Laurito et al., Xu et al., RepreGuard, and Quaremba et al.
- Position: source recognizability and preference do not show that prompt authorship style controls task behavior.

### Evaluation awareness and behavior

- Needham et al.; Abdelnabi and Salem; Nguyen et al.; Li et al.; Nayan et al.; Heidari et al.
- Position: studies motivate the recognition/behavior/control split; our manipulation is authorship style rather than evaluation framing.

### Prompt construction confounds

- Devbunova; Chalkidis; EvalDetectBench evidence.
- Position: motivates content matching, surface adjustment, paraphrase controls, generator metadata, and random steering directions.

## 3. Methodology

### Problem formulation

- For item i, responder m, condition c, define binary outcome Y and paired risk difference E[Y_llm - Y_human].
- Primary family: harmful compliance, sycophancy, and accuracy across two API responders.

### Design overview

- Figure: source items -> controlled variants -> blinded validation -> responder outcomes -> paired statistics; separate Qwen probe/steering branch.

### Data and rewrite construction

- 120 fixed items, 30 per task; identify datasets and binary endpoints.
- Both controlled variants are machine-generated; interpret the intervention as perceived style, not literal provenance.
- Fidelity threshold >=4/5 and no substantive addition/omission; six unusable pairs.

### Models, scoring, and controls

- Give exact IDs and Qwen revision.
- Deterministic option parsing for accuracy/sycophancy; blinded rubric for harmful/benign tasks.
- Original-prompt baseline, covariates, crossed source labels, surface classifier, no-steer and random directions.

### Statistical analysis

- Paired bootstrap (10,000), exact discordant-pair tests, Holm over six primary tests.
- Task-specific linear probability models with item-clustered standard errors; exclude constant covariates.
- State success criteria and avoid equivalence interpretation.

### White-box protocol

- 105 complete pairs, grouped 126/42/42 split, every-fourth-layer sweep, validation selection at layer 4.
- Evaluation-awareness direction from 80 paired examples; 16 fixed prompts; +/-1 projection SD; five random directions; greedy 64-token generation.

## 4. Results

### Manipulation check

- Figure: `authorship_manipulation.png`.
- Table: task-wise paired AI-authorship shifts and bootstrap intervals.
- Finding: positive in every task, fidelity 4.79--5.00.

### Primary behavior

- Figure: `paired_effects.png`.
- Standalone table: all eight task/model contrasts, exact and Holm p-values.
- Finding: none of six primary tests significant; sycophancy is the largest common signal but remains inconclusive.

### Surface and source controls

- Table: adjusted pooled risk differences.
- Discuss original-to-human paraphrase exception and mostly null explicit-label subset.

### Representation and intervention

- Standalone table: AUROC, balanced accuracy, surface baseline, cosine, logit shift.
- Figure: `steering_outcomes.png`.
- Finding: strong decodability, modest overlap with evaluation awareness, random-like behavioral steering.

### Missingness and failure modes

- 114 successful pairs, 625 unique scores, 97.1% validity; selective harmful-rewrite failures and floor/ceiling behavior.

## 5. Discussion

- Interpret hypotheses H1--H6 without overstating nulls.
- Main implication: synthetic prompt transfer receives cautious reassurance only in this bounded setting.
- Mechanistic implication: probe AUROC cannot establish causal use.
- Limitations: machine-produced human-like variants, pilot precision, judge dependence, model scope, rewrite-template dependence, steering size/layer/sample, evaluation proxy.
- Evidence quality: low-to-moderate for the bounded null, low for deployment-wide absence.
- Broader impact/ethics: improved evaluation validity; intervention methods are dual-use; harmful prompts retained only for controlled audit.

## 6. Conclusion

- Restate paired multi-behavior audit and white-box stress test.
- Key takeaway: recognition is clear, stable downstream use is not.
- Future work: 150--300 pairs per task/model, genuine multi-annotator human rewrites, multiple generators/model families, borderline safety items, equivalence margins, broader layer/token/magnitude search.

## Appendix plan

- Endpoint definitions and success criteria.
- Full model and resource accounting.
- Reproducibility artifact paths and hashes.

## Figure and table plan

| ID | Type | Location | Content |
|---|---|---|---|
| Fig. 1 | Diagram | Methodology | Experimental and inferential pipeline |
| Fig. 2 | Plot | Results | Perceived AI-authorship manipulation |
| Fig. 3 | Plot | Results | Paired API behavior effects |
| Fig. 4 | Plot | Results | Steering versus random controls |
| Tab. 1 | Table | Methodology | Tasks, datasets, endpoints |
| Tab. 2 | Table | Results | Manipulation effects |
| Tab. 3 | Table | Results | Paired primary and secondary effects |
| Tab. 4 | Table | Results | Surface-adjusted effects |
| Tab. 5 | Table | Results | White-box probe and intervention summary |

## Citation checklist

- Evaluation recognition: Needham et al. (2025)
- Test-awareness intervention: Abdelnabi and Salem (2025); Nguyen et al. (2025)
- Representation/behavior dissociation: Heidari et al. (2026); Nayan et al. (2026); Li et al. (2026)
- Human/machine representations: Chen et al. (2025); Quaremba et al. (2026)
- Source preference: Panickssery et al. (2024); Laurito et al. (2024); Xu et al. (2025)
- Construction confounds: Devbunova (2026); Chalkidis (2026)
- Datasets/models: JBB-Behaviors, MMLU, Anthropic model-written evals, Qwen3 technical report/model card as needed.
