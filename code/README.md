# Cloned repositories

All 11 repositories were inspected at the recorded commit. No full model run was attempted because the main pipelines require model weights, GPUs, or paid API credentials; entry points and data assets were validated by inspection.

| Repository | Commit | Purpose | Key entry points / notes |
|---|---:|---|---|
| [lauritowal/ai-ai-bias](https://github.com/lauritowal/ai-ai-bias) | `73475012` | Reproduces human-vs-LLM presentation preference experiments | `scripts/generate_llm_descriptions.py`, `scripts/run_comparisons.py`; includes source data, cached outputs, and analysis results. |
| [jjpn97/eval_awareness](https://github.com/jjpn97/eval_awareness) | `1543ffe` | Black-box evaluation/deployment classification and calibration | `run.py --mini`; uses Inspect and an external dataset path; API keys required. |
| [Jordine/evaluation-awareness-probing](https://github.com/Jordine/evaluation-awareness-probing) | `5f901bf` | Minimal Llama-3.3-70B probe and SAE-steering implementation | `scripts/generate_vectors.py`, `scripts/analyze_probe.py`; GPU and model weights required. |
| [microsoft/Test_Awareness_Steering](https://github.com/microsoft/Test_Awareness_Steering) | `eb78ea5` | Probe training, parameter editing, and harmful action evaluation | Includes 2,587 trigger pairs, annotated traces, and action data; primary reusable white-box baseline. |
| [NLP2CT/RepreGuard](https://github.com/NLP2CT/RepreGuard) | `53677be` | Representation-based human/LLM text detector | `repreGuard_detector.py`, `repreGuard_evaluation.py`; external DetectRL/RAID data and model weights required. |
| [gerritq/mgt_probes](https://github.com/gerritq/mgt_probes) | `8875297` | Layer-wise and concatenated-layer machine-text probes | `src/probes/probe_main.py`, `probe.sh`, `baseline.sh`; best starting point for authorship-probe implementation. |
| [viliana-dev/eval-awareness-format](https://github.com/viliana-dev/eval-awareness-format) | `b6a5000` | Controlled 2×2 format/context study | `scripts/prepare_datasets.py`, `rewrite_to_benchmark.py`, `evaluate_probe_cached.py`; directly reusable confound controls. |
| [aisa-group/decomposing-eval-awareness](https://github.com/aisa-group/decomposing-eval-awareness) | `2961c17` | EvalAwareBench runners and recognition/propensity analysis | Sparse checkout retains code, scripts, data interfaces, and docs; full benchmark available separately on Hugging Face. |
| [evaluation-awareness/evaluation-awareness](https://github.com/evaluation-awareness/evaluation-awareness) | `8ad6a8e` | Minimal reproduction of probe AUROC, correlation, mutual information, and random-direction controls | `cls_corr_mi/reproduce.sh`; README provides a CPU smoke test and six-model workflow. |
| [younwoochoi/InterlocutorAwarenessLLM](https://github.com/younwoochoi/InterlocutorAwarenessLLM) | `1edb852` | Model-family inference from reasoning, linguistic style, and alignment preferences | `unified_response_generator.py`, `unified_evaluation.py`; includes cooperative, reward-hacking, and jailbreak case-study code. |
| [freeze-lasr/aware_bench](https://github.com/freeze-lasr/aware_bench) | `8e49d16` | EvalDetectBench collection, per-judge prompt calibration, and generator-confound correction | `run.py`, `configs/config_test.yaml`; integrates Inspect logs and Hugging Face deployment sources. |

## Recommended reuse order

1. Adapt `ai-ai-bias` for randomized paired behavioral comparisons.
2. Reuse the orthogonalized 2×2 construction in `eval-awareness-format` for source-style and surface-feature controls.
3. Use `mgt-probes` for perceived-authorship probes and `evaluation-awareness-representation` for probe/judge correlation and random-direction controls.
4. Use `interlocutor-awareness` for source-identity behavioral adaptations and `evaldetectbench` to control generator identity and model-specific elicitation.
5. Use `Test_Awareness_Steering` only after a robust behavioral effect is established; steering is a causal stress test, not the primary evidence.

## Practical requirements

- Closed-model runs need provider API keys and a pinned model/version log.
- White-box runs need Hugging Face model access and typically an A100-class GPU for 7B–32B models.
- Several repositories pin their own requirements. Do not install them wholesale into the workspace environment; add only the packages needed by the chosen experiment to `pyproject.toml`.
- Preserve the current project `.venv`; repository-specific Conda instructions should not be used.
