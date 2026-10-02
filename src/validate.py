"""Fail-fast validation of final research artifacts and scientific invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    required = [
        "REPORT.md", "README.md", "planning.md", "pyproject.toml", "uv.lock",
        "results/data/base_items.jsonl", "results/evaluations/paired_primary.csv",
        "results/evaluations/analysis_summary.json", "results/evaluations/probe_results.json",
        "results/evaluations/steering_controls.csv", "figures/authorship_manipulation.png",
        "figures/paired_effects.png", "figures/steering_outcomes.png",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    assert not missing, f"Missing artifacts: {missing}"

    items = jsonl(ROOT / "results/data/base_items.jsonl")
    assert len(items) == 120 and len({x["item_id"] for x in items}) == 120
    assert {x["task"] for x in items} == {"accuracy", "sycophancy", "harmful_compliance", "benign_overrefusal"}

    scores = jsonl(ROOT / "results/evaluations/behavior_scores.jsonl")
    unique_scored = {(x["item_id"], x["style"], x["responder"]) for x in scores if "error" not in x and x.get("outcome") is not None}
    assert len(unique_scored) == 625
    primary = pd.read_csv(ROOT / "results/evaluations/paired_primary.csv")
    assert len(primary) == 8 and primary.risk_difference.notna().all()
    assert len(primary[primary.task.isin(["accuracy", "sycophancy", "harmful_compliance"])]) == 6

    probe = json.loads((ROOT / "results/evaluations/probe_results.json").read_text())
    assert probe["test_auc"] > 0.7 and probe["steering_scored"] == 128
    assert len(probe["expected_probe_logit_shifts_random"]) == 5

    report = (ROOT / "REPORT.md").read_text()
    for section in ["Executive Summary", "Experimental Setup", "Results", "Limitations", "Conclusions"]:
        assert section in report
    for figure in ["figures/authorship_manipulation.png", "figures/paired_effects.png", "figures/steering_outcomes.png"]:
        assert figure in report

    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in required}
    result = {
        "status": "passed", "required_artifacts": len(required), "base_items": len(items),
        "unique_scored_behavior_rows": len(unique_scored), "primary_contrasts": len(primary),
        "probe_test_auc": probe["test_auc"], "steering_scores": probe["steering_scored"],
        "artifact_sha256": hashes,
    }
    (ROOT / "results/evaluations/validation_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
