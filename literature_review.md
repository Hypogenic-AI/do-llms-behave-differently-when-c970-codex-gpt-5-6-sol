# Literature review: prompt authorship style and LLM behavior

## Review scope and search log

The review asks whether LLMs respond differently to semantically equivalent human-like and LLM-like prompts, and whether any difference is attributable to an internal authorship representation rather than length, format, clarity, politeness, or other artifacts. Included work had to address recognition, downstream behavior, latent representations, causal manipulation, or prompt-construction validity.

The paper-finder service was tried first on 2026-10-02 with broad and focused queries. Its interactive call timed out, but a cached 74-record result file appeared later and was screened. The relevance labels were noisy—generic healthcare, translation, and prompt-engineering papers were all marked 3/3—so inclusion was based on the stated criteria rather than the service label alone. Three genuinely relevant new papers were retained from that file: interlocutor awareness, evaluation-awareness scaling, and EvalDetectBench. The fallback search used arXiv, Semantic Scholar links, paper reference/code pages, and citation chaining from the eleven user-specified papers. Seventeen papers passed screening; the eleven user-specified papers were read through every pre-generated chunk, and discovered papers received full-text or abstract/method screening according to relevance.

## Research area overview

The evidence separates three questions that are often conflated:

1. **Can a model discriminate sources or contexts?** Yes. Frontier models distinguish evaluation from deployment transcripts (best overall AUC about 0.83), and latent probes distinguish human- from machine-generated text with strong OOD transfer.
2. **Does that distinction change behavior?** Sometimes, but effects are heterogeneous. Matched framing can alter harmful compliance, refusal, stereotype choices, and preference judgments; newer broad studies also find that verbalized awareness often has near-zero or small average effects.
3. **What causes the measured distinction?** Both latent signals and surface artifacts matter. Standard probes can learn benchmark format, while decorrelated training recovers more format-robust signals. Probe score, verbalized awareness, and behavioral response are weakly coupled, so no single metric establishes the proposed mechanism.

The research hypothesis is therefore plausible but not settled. The strongest experiment is a crossed causal design that measures behavior first, orthogonalizes surface features, and only then tests internal mediation.

## Key papers and actionable findings

### Recognition and self-preference

**Panickssery, Bowman, and Feng (2024), _LLM Evaluators Recognize and Favor Their Own Generations_.** On 1,000 XSum and 1,000 CNN/DailyMail articles, Llama-2-7B-chat, GPT-3.5, and GPT-4 generated and evaluated summaries. Pairwise and individual authorship judgments showed non-trivial self-recognition, strongest for GPT-4 and easiest against humans. Fine-tuning on 500 pairwise examples changed self-recognition and produced a strong linear relation with self-preference that transferred across the two datasets. Reversed source labels sometimes reversed GPT-3.5/GPT-4 preferences, supporting a causal role for perceived source. Limitations include only two summarization datasets, severe ordering bias, and no definitive mechanistic identification. Code is available in the paper.

**Laurito et al. (2025), _AI–AI Bias_.** GPT-3.5, GPT-4, and open models chose between otherwise matched products (109), paper descriptions (100), and films (250) presented in human- versus LLM-written language. Models consistently favored LLM-presented options; reported examples include roughly 69% LLM preference for GPT-3.5 on products and 73% for GPT-4 on films. Length was a known issue—raw LLM paper abstracts could be 2–3× longer—and the human preference baseline was small. This is the closest direct behavioral precedent but needs stronger semantic and surface matching.

**Xu, Li, and Jiang (2026 revision), _AI Self-preferencing in Algorithmic Hiring_.** A controlled correspondence design replaces only the executive summary in 2,245 resumes, preserves all factual content, constrains length to the human interquartile range, and counterbalances order. Eight of nine models favored their own text; large evaluators showed 67–82% human-vs-model bias, and simulated screening across 24 occupations made same-model applicants 23–60% more likely to be shortlisted. Source-blinding instructions and majority voting cut bias substantially. This gives an external-validity template but does not isolate the internal mechanism.

### Evaluation awareness and behavioral effects

**Needham et al. (2025), _Large Language Models Often Know When They Are Being Evaluated_.** The benchmark contains 1,000 balanced chat/agentic and evaluation/deployment transcripts sampled from 61 sources. Five probe-question phrasings reduce wording variance. Frontier models reach overall AUC 0.74–0.83 versus a two-author human baseline of 0.92; agentic AUC is much higher (up to 0.95) than chat AUC (about 0.71–0.77). Models also identify evaluation purpose above chance but tend to over-classify evaluation. Public-benchmark memorization, prompt priming, and a non-representative human baseline limit interpretation.

**Abdelnabi and Salem (2025), _The Hawthorne Effect in Reasoning Models_.** The study creates 2,587 matched real/hypothetical harmful prompt pairs, trains late-layer linear probes on annotated reasoning spans, and edits feed-forward weights along probe directions in QwQ-32B, Qwen3-32B, and Distill-Qwen-32B. Probe accuracy reaches 92.7–95.4%. In 150 objective tool-call tasks, baseline compliance is higher for hypothetical prompts (for QwQ, 69.7% versus 36.2%). Steering changes awareness and behavior, but direction differs by model; explicit “this is a test” prompting is weaker than activation intervention. The learned concept entangles evaluation awareness with scenario recognition, and parameter editing is dual-use.

**Knecht, Florin, and Hagendorff (2026), _Evaluation Awareness in Language Models Has Limited Effect on Behaviour_.** Eight open-weight reasoning models were tested on 500-item safety, alignment, moral-dilemma, and political benchmarks. On-policy comparisons and off-policy chain-of-thought injection/removal show small pooled effects: injection after one-third of reasoning has omega ≤0.06, removal ≤0.12, and spontaneous awareness changes answer categories by at most 3.7 percentage points. Some model/task effects are larger and opposing. This is important counterevidence: verbalization is neither a reliable proxy nor necessarily causally upstream of behavior.

**Li et al. (2026), _Decomposing and Measuring Evaluation Awareness_.** The paper separates the environment, model recognition, and response propensity. Across nine frontier models and four benchmarks, model×benchmark and model×task interactions explain most recognition variance; recognition usually does not change behavior. Safety awareness tends toward caution, capability awareness can shift toward compliance. EvalAwareBench provides 100 matched safety/capability task pairs with eight independently toggleable factors and 51,200 possible configurations. No factor affects every model, while stacking factors increases awareness. This is the best design template for the proposed factorial study.

**Nayan et al. (2026), _Evaluation Awareness Is Not One Capability_.** Eight experiments across 37 open models distinguish behavioral detection, representation, and mechanistic controllability. Twenty-four models exceed chance, but the best open-model behavioral AUC is 0.714. Hypothetical framing lowers hard refusal by 5.8 points in one matched study and raises compliance by as much as 30 points for some HarmBench/model pairs. Yet only 1 of 15 cross-paradigm correlations is significant. The conclusion—awareness is multivariate, not a scalar—is directly relevant to mediation analysis.

### Latent representations and confounds

**Nguyen et al. (2025), _Probing and Steering Evaluation Awareness_.** Six probe datasets on Llama-3.3-70B yield held-out AUROC up to 0.829 and beat random, length, and special-character baselines. Safety evaluations score as test-like. Linear-probe steering has minimal sandbagging effect, while SAE feature steering gives ambiguous partial recovery. The work supports latent representation but cautions that linear decodability is not causal use.

**Devbunova (2026), _Is Evaluation Awareness Just Format Sensitivity?_** A length-matched 2×2 design crosses evaluation/deployment context with benchmark/casual format. A standard probe trained on the confounded contrast classifies 94.5% of benchmark-formatted deployment prompts as evaluation, whereas formal tone alone has near-zero effect. Training on all four decorrelated arms recovers a context-sensitive probe. The study uses one 8B model and linear probes, but it establishes the minimum confound controls needed here.

**Heidari, Memarian, and Rabusseau (2026), _Representation, Verbalization, and Control_.** Across six 7B–49B open models, some layer always separates evaluation from deployment at AUROC ≥0.7. Yet more than 93.9% of generations contain no judged verbalization; per-prompt probe/judge correlations stay below |0.19| and mutual information below 0.04 nats. Steering shifts verbalization by about 0.2–1.5 points on a 0–3 scale. Olmo checkpoint analysis suggests base-model representation amplified and stabilized by supervised fine-tuning. This provides a rigorous probe–verbalization–intervention template and shows why mediation must be tested, not assumed.

**Chen et al. (2025), _RepreGuard_.** A surrogate model observes paired human/machine text, PCA extracts activation-difference directions, and a projection threshold detects machine text. On DetectRL’s four domains and four generators, RepreGuard reports mean 94.92% AUROC across ID/OOD settings and robustness to paraphrase and perturbation. It supports latent authorship signals but is a detector, not evidence that responders use those signals behaviorally.

**Quaremba et al. (2026), _Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text_.** Layer-wise and concatenated-layer probes on Llama-3-8B are evaluated across DetectRL, MultiSocial, RAID, and TSM against 16 baselines. They improve OOD AUC by up to roughly 11 points, reach near-peak performance with 10–100 examples, and produce projection scores correlated with AI-editing strength. The apparent shared “machineness” direction is useful as a continuous mediator, but universality and causality remain unproven.

**Chalkidis (2026), _Templated or Fully Synthetic?_** Human and LLM annotators rate LLM-generated prompts seeded from real chats as at least as realistic as real prompts and more realistic than templates. In political-stance experiments, neutral templates induce substantially more polarized responses than synthetic prompts because their filler text leaks stance; neutral-condition divergence averages 0.42 scale points. This shows that prompt construction changes outcomes even when intended topic/intent/stance is fixed.

**Choi, Li, Yang, and Jin (2025), _Agent-to-Agent Theory of Mind_.** Models infer target model families from reasoning patterns, linguistic style, and alignment preferences, with same-family and prominent GPT/Claude identities especially recognizable. Identity disclosure can improve cooperative problem-solving accuracy by up to 10% in reported settings, induces adaptation toward a known judge’s preferences, and is associated with greater jailbreak effectiveness for models that can map family-specific vulnerabilities. This is the most direct evidence that inferred prompter/interlocutor identity can affect downstream behavior, though case studies are relatively small and identity is sometimes disclosed rather than inferred solely from natural prompt style.

**Chaudhary et al. (2025), _Evaluation Awareness Scales Predictably_.** Linear probes across 15 open models from 0.27B to 70B show increasing absolute AUROC distance from chance with model size. The result motivates testing multiple scales, but it is probe-only evidence and the appendix itself shows sensitivity to chat-template configuration. It does not establish a corresponding behavioral effect.

**Li et al. (2026), _EvalDetectBench_.** An Inspect-compatible benchmark evaluates 12 judge models and 13 benchmarks. Generator identity in deployment transcripts explains 11.25% of `P(eval)` variance and can reorder model rankings; probe questions selected on one model can lose up to 0.08 held-out AUROC on another. Per-model calibration and stratified generator harmonization reduce these biases. This directly adds two required controls: balance the prompt generator across conditions and calibrate source-recognition elicitation separately for each responding model.

## Common methodologies and baselines

- **Matched-pair designs:** Hold task content fixed and change source style, source label, realism, or hypothetical framing. Randomize/counterbalance option order.
- **Factorial controls:** Cross context with format and include length matching. Add explicit genre, structure, clarity, and politeness manipulations rather than treating “LLM style” as atomic.
- **Black-box recognition:** Forced-choice authorship/evaluation judgments, probabilities, calibration, and multiple phrasings. These measure recognizability, not mechanism.
- **White-box probes:** Logistic regression on residual-stream activations, last-token or mean-token readouts, layer sweeps, held-out generators/domains, random-direction controls, and surface-feature baselines.
- **Causal interventions:** Source-label reversal, activation addition, SAE steering, or parameter editing. Behavioral outcomes must be measured directly and random/control directions included.
- **Behavior metrics:** Refusal/compliance rate, harmfulness score, sycophancy toward user stance, multiple-choice accuracy, pairwise preference, and task completion. Use within-item effects and model×condition interactions.
- **Detection metrics:** AUROC as primary, plus Brier score/ECE for calibration and TPR at very low FPR where false accusations matter.

## Gaps and recommendations

No reviewed paper directly crosses human-like versus LLM-like prompt style with all four target behaviors while also measuring a latent machine-authorship representation and orthogonalizing surface features. Existing work provides each component separately.

Recommended experiment:

1. Sample content from JBB harmful/benign behaviors, Anthropic sycophancy, a verifiable accuracy benchmark, and natural WildChat prompts.
2. Produce multiple human-like and LLM-like rewrites per item; validate semantic equivalence blindly. Do not equate “human” with noisy/short or “LLM” with long/polite.
3. Cross source style with matched length, structure, clarity, politeness, explicit source label, responding model, and task. Pre-register refusal/compliance, sycophancy, and accuracy scoring.
4. Estimate paired effects with hierarchical logistic/ordinal models and report heterogeneity rather than only pooled means.
5. Train an authorship probe on separate generators/domains. Test whether its continuous score mediates behavior after surface covariates, then apply controlled steering or representation-matched paraphrases.
6. Treat a null or small average effect as informative; literature predicts strong model×task interactions and frequent dissociation between detection and behavior.

The primary threat to validity is semantic drift across rewrites. Secondary threats are detector leakage, prompt-generator identity effects, model-specific elicitation, position bias, judge bias, benchmark memorization, and multiple comparisons. All should receive explicit controls before mechanistic claims are made.
