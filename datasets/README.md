# Downloaded datasets

The data files are available locally for the experiment runner but excluded from Git by `datasets/.gitignore`. Small sanitized examples are retained in `datasets/samples/`.

## 1. WildChat-1M sample

- **Source:** `allenai/WildChat-1M`
- **Local location:** `datasets/wildchat_5k/`
- **Local size:** 5,000 rows, about 19 MiB on disk
- **Format:** Hugging Face `Dataset` saved to Arrow
- **Schema:** `conversation_hash`, `model`, `timestamp`, `conversation`, `turn`, `language`, moderation fields, redaction/toxicity flags, and coarse client metadata
- **Use:** Natural human/deployment prompt seeds and a negative class for authorship/evaluation probes
- **Validation:** Loaded with `load_from_disk`; 5,000/5,000 rows readable. The included sample is sanitized and omits direct identifiers.
- **License:** The local dataset card does not declare a license; downstream use must follow the source dataset terms and privacy guidance.

Download/recreate:

```python
from datasets import load_dataset

stream = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
sample = stream.take(5000)
# Materialize after applying the project's sanitization policy, then save_to_disk.
```

Load locally:

```python
from datasets import load_from_disk
wildchat = load_from_disk("datasets/wildchat_5k")
```

## 2. JBB-Behaviors

- **Source:** `JailbreakBench/JBB-Behaviors`
- **Local location:** `datasets/jbb_behaviors/`
- **Local size:** about 873 KiB
- **Format:** CSV plus dataset card and generation scripts
- **Splits/content:** 100 harmful behaviors, 100 matched benign behaviors, and 300 judge-comparison rows
- **Schema:** harmful/benign files contain `Index`, `Goal`, `Target`, `Behavior`, `Category`, `Source`
- **Use:** Primary refusal/harmful-compliance items. The benign set is a negative control.
- **License:** MIT
- **Validation:** Every CSV parsed successfully with the expected columns and row counts.

Download:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="JailbreakBench/JBB-Behaviors",
    repo_type="dataset",
    local_dir="datasets/jbb_behaviors",
)
```

Load:

```python
import pandas as pd
harmful = pd.read_csv("datasets/jbb_behaviors/data/harmful-behaviors.csv")
benign = pd.read_csv("datasets/jbb_behaviors/data/benign-behaviors.csv")
```

## 3. Anthropic model-written evaluations

- **Source:** `Anthropic/model-written-evals`
- **Local location:** `datasets/anthropic_model_written_evals/`
- **Local size:** about 21 MiB allocated locally
- **Format:** JSONL plus dataset cards
- **Validated contents:** 30,168 sycophancy items; 98,362 persona items across 99 files; 16,400 LM-generated and 8,116 human-generated advanced-risk items; 3,289 WinoGenerated records
- **Use:** Sycophancy outcomes; human-versus-model-written evaluation controls; optional self-awareness/persona tasks
- **License:** The dataset card lists the arXiv perpetual non-exclusive license rather than a standard permissive software license; confirm redistribution conditions before publishing derivatives.

Download:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Anthropic/model-written-evals",
    repo_type="dataset",
    local_dir="datasets/anthropic_model_written_evals",
)
```

Load one sycophancy task:

```python
import json
path = "datasets/anthropic_model_written_evals/sycophancy/sycophancy_on_nlp_survey.jsonl"
with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
```

## 4. EvalDetectBench paper-replication subset

- **Source:** `el7982/aware-bench`
- **Local location:** `datasets/evaldetectbench_replication/`
- **Local size:** about 2.1 MiB
- **Format:** CSV replication tables plus dataset documentation
- **Content:** 11 validated tables, including 19,034 scored measurement rows, 21,887 combined scored rows, generator/judge cross-products, depth-resampling results, and per-model probe-selection AUROCs
- **Use:** Generator-identity confound analysis, black-box source/evaluation recognition, and probe-question calibration
- **Download decision:** The full repository is about 1.9 GB and is dominated by raw model logs. The compact paper-replication subset was downloaded under the large-dataset rule; the full trajectories remain reproducible on demand.
- **License:** Confirm the current Hugging Face dataset card and source-dataset terms before redistribution; constituent transcript sources have heterogeneous licenses.

Download the staged subset:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="el7982/aware-bench",
    repo_type="dataset",
    local_dir="datasets/evaldetectbench_replication",
    allow_patterns=["README.md", "paper_replication/*"],
)
```

To retrieve the entire 1.9 GB artifact, remove `allow_patterns`.

## Code-bundled experimental data

The associated repositories also include immediately usable data:

- `code/test-awareness-steering/data/triggers/triggers.json`: 2,587 real/hypothetical pairs.
- `code/test-awareness-steering/data/actions/test_actions_examples.json`: objective tool-call compliance items.
- `code/evaluation-awareness-probing/datasets/`: contrastive probe training and validation JSON files.
- `code/decomposing-eval-awareness/`: loaders and runners for EvalAwareBench; the full benchmark is downloadable from `aisa-group/EvalAwareBench` under CC BY-NC 4.0.

These are documented as code assets rather than counted among the three primary dataset packages.

## Small samples

- `samples/wildchat_sanitized_examples.json`
- `samples/harmful-behaviors.csv`
- `samples/benign-behaviors.csv`
- `samples/anthropic_self_awareness_examples.json`

The samples are for schema inspection only and are not statistically representative.
