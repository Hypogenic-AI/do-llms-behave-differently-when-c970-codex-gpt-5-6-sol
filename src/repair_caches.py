"""Remove only failed generated cache rows so the harness can retry them safely."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    ROOT / "results/model_outputs/rewrites.jsonl",
    ROOT / "results/evaluations/rewrite_validation.jsonl",
    ROOT / "results/model_outputs/behavior_responses.jsonl",
    ROOT / "results/evaluations/behavior_scores.jsonl",
    ROOT / "results/model_outputs/label_responses.jsonl",
    ROOT / "results/evaluations/label_scores.jsonl",
]


def main() -> None:
    for path in PATHS:
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        good = [row for row in rows if "error" not in row]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in good))
        tmp.replace(path)
        print(f"{path.relative_to(ROOT)}: retained {len(good)}/{len(rows)}; retrying {len(rows)-len(good)}")


if __name__ == "__main__":
    main()
