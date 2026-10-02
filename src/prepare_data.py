"""Create the fixed, stratified pilot sample and descriptive data audit."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data" / "base_items.jsonl"
SEED = 20261002
N_PER_TASK = 30


def stratified_rows(frame: pd.DataFrame, group: str, n: int, seed: int) -> pd.DataFrame:
    """Take a reproducible roughly proportional sample while retaining groups."""
    rng = random.Random(seed)
    groups = {str(k): list(v.index) for k, v in frame.groupby(group)}
    chosen: list[int] = []
    keys = sorted(groups)
    while len(chosen) < n:
        progressed = False
        for key in keys:
            candidates = [i for i in groups[key] if i not in chosen]
            if candidates and len(chosen) < n:
                chosen.append(rng.choice(candidates))
                progressed = True
        if not progressed:
            break
    return frame.loc[chosen].copy()


def load_sycophancy() -> list[dict]:
    path = ROOT / "datasets/anthropic_model_written_evals/sycophancy/sycophancy_on_nlp_survey.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rng = random.Random(SEED + 2)
    # Balance which option matches the user's stated belief.
    by_answer: dict[str, list[dict]] = {"A": [], "B": []}
    for row in rows:
        label = row["answer_matching_behavior"].strip().strip("()")
        if label in by_answer:
            by_answer[label].append(row)
    selected = rng.sample(by_answer["A"], N_PER_TASK // 2) + rng.sample(
        by_answer["B"], N_PER_TASK - N_PER_TASK // 2
    )
    rng.shuffle(selected)
    out = []
    for i, row in enumerate(selected):
        out.append(
            {
                "item_id": f"syc_{i:03d}",
                "task": "sycophancy",
                "source_prompt": row["question"],
                "target_choice": row["answer_matching_behavior"].strip().strip("()"),
                "non_target_choice": row["answer_not_matching_behavior"].strip().strip("()"),
                "source": "Anthropic/model-written-evals:nlp_survey",
            }
        )
    return out


def format_mcq(row: dict) -> str:
    letters = "ABCD"
    options = "\n".join(f"({letters[i]}) {value}" for i, value in enumerate(row["choices"]))
    return f"{row['question']}\n\nChoices:\n{options}\n\nAnswer with only the letter:"


def load_accuracy() -> list[dict]:
    dataset = load_dataset("cais/mmlu", "all", split="test")
    frame = dataset.to_pandas()
    selected = stratified_rows(frame, "subject", N_PER_TASK, SEED + 3)
    out = []
    for i, (_, row) in enumerate(selected.iterrows()):
        raw = row.to_dict()
        out.append(
            {
                "item_id": f"acc_{i:03d}",
                "task": "accuracy",
                "source_prompt": format_mcq(raw),
                "correct_choice": "ABCD"[int(raw["answer"])],
                "subject": raw["subject"],
                "source": "cais/mmlu:all:test",
            }
        )
    return out


def load_jbb(kind: str, seed: int) -> list[dict]:
    path = ROOT / f"datasets/jbb_behaviors/data/{kind}-behaviors.csv"
    frame = pd.read_csv(path)
    selected = stratified_rows(frame, "Category", N_PER_TASK, seed)
    task = "harmful_compliance" if kind == "harmful" else "benign_overrefusal"
    return [
        {
            "item_id": f"{'harm' if kind == 'harmful' else 'benign'}_{i:03d}",
            "task": task,
            "source_prompt": row.Goal,
            "category": row.Category,
            "behavior": row.Behavior,
            "source": f"JBB-Behaviors:{row.Source}",
            "jbb_index": int(row.Index),
        }
        for i, (_, row) in enumerate(selected.iterrows())
    ]


def main() -> None:
    random.seed(SEED)
    items = (
        load_jbb("harmful", SEED)
        + load_jbb("benign", SEED + 1)
        + load_sycophancy()
        + load_accuracy()
    )
    assert len(items) == 4 * N_PER_TASK
    assert len({x["item_id"] for x in items}) == len(items)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))

    audit = pd.DataFrame(
        {
            "task": [x["task"] for x in items],
            "words": [len(x["source_prompt"].split()) for x in items],
            "chars": [len(x["source_prompt"]) for x in items],
            "missing_prompt": [not bool(x["source_prompt"].strip()) for x in items],
            "duplicate_prompt": [
                sum(y["source_prompt"] == x["source_prompt"] for y in items) > 1 for x in items
            ],
        }
    )
    summary = {
        "seed": SEED,
        "n": len(items),
        "task_counts": audit.task.value_counts().to_dict(),
        "missing_prompts": int(audit.missing_prompt.sum()),
        "duplicate_prompts": int(audit.duplicate_prompt.sum()),
        "word_count_by_task": audit.groupby("task").words.agg(["mean", "std", "min", "max"]).round(2).to_dict("index"),
    }
    (OUT.parent / "data_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
