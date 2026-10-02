"""Summarize fixed-text activation steering against norm-matched random controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "results/evaluations/steering_scores.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [x for x in rows if "error" not in x and x.get("outcome") is not None]
    frame = pd.DataFrame(
        {"item_id": x["item_id"], "task": x["task"], "condition": x["steering_condition"],
         "outcome": int(x["outcome"]), "valid": bool(x.get("valid", True)),
         "response": x.get("response", ""), "score": x.get("score")}
        for x in rows
    ).drop_duplicates(["item_id", "condition"], keep="last")
    agg = frame.groupby(["task", "condition"]).agg(n=("outcome", "size"), outcome_rate=("outcome", "mean"), valid_rate=("valid", "mean"), mean_score=("score", "mean")).reset_index()
    agg.to_csv(ROOT / "results/evaluations/steering_aggregate.csv", index=False)

    base = frame[frame.condition == "none"].set_index("item_id")
    changes = []
    for cond, group in frame[frame.condition != "none"].groupby("condition"):
        current = group.set_index("item_id").loc[base.index]
        changes.append(
            {"condition": cond, "n": len(base),
             "outcome_change": float((current.outcome - base.outcome).mean()),
             "validity_change": float((current.valid.astype(int) - base.valid.astype(int)).mean()),
             "response_changed_rate": float((current.response != base.response).mean())}
        )
    change = pd.DataFrame(changes)
    change.to_csv(ROOT / "results/evaluations/steering_controls.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=agg, x="task", y="outcome_rate", hue="condition")
    plt.ylabel("Outcome rate (risk; accuracy for MMLU)"); plt.xlabel(""); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(ROOT / "figures/steering_outcomes.png", dpi=180); plt.close()

    probe = json.loads((ROOT / "results/evaluations/probe_results.json").read_text())
    probe["steering_summary"] = {
        "author_pos_outcome_change": float(change.loc[change.condition == "author_pos", "outcome_change"].iloc[0]),
        "author_neg_outcome_change": float(change.loc[change.condition == "author_neg", "outcome_change"].iloc[0]),
        "random_outcome_change_range": [float(change[change.condition.str.startswith("random")].outcome_change.min()), float(change[change.condition.str.startswith("random")].outcome_change.max())],
        "author_pos_validity_change": float(change.loc[change.condition == "author_pos", "validity_change"].iloc[0]),
        "author_neg_validity_change": float(change.loc[change.condition == "author_neg", "validity_change"].iloc[0]),
    }
    (ROOT / "results/evaluations/probe_results.json").write_text(json.dumps(probe, indent=2))
    print(agg.to_string(index=False)); print("\n", change.to_string(index=False))


if __name__ == "__main__":
    main()
