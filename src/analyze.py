"""Paired statistical analysis, robustness checks, and publication figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm
import textstat

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/evaluations"
RAW = ROOT / "results/model_outputs"
FIG = ROOT / "figures"
SEED = 20261002


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten_scores(path: Path) -> pd.DataFrame:
    rows = []
    for x in read(path):
        if "error" in x or x.get("outcome") is None:
            continue
        v = x.get("validation", {}) or {}
        rows.append(
            {
                "item_id": x["item_id"], "task": x["task"], "style": x["style"],
                "responder": x["responder"], "outcome": int(x["outcome"]),
                "score": x.get("score"), "explicit_refusal": x.get("explicit_refusal"),
                "valid": bool(x.get("valid", True)), "prompt": x["prompt"],
                "response": x.get("response", ""), "words": len(x["prompt"].split()),
                "readability": textstat.flesch_reading_ease(x["prompt"]),
                "clarity": v.get("clarity"), "politeness": v.get("politeness"),
                "p_ai": v.get("p_ai"), "fidelity": v.get("fidelity"),
                "generator": (x.get("api", {}) or {}).get("model_returned"),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean(diff: np.ndarray, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    if len(diff) == 0:
        return np.nan, np.nan
    vals = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)])
    return tuple(np.quantile(vals, [0.025, 0.975]))


def paired_table(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    rows = []
    for (task, responder), group in df[df["style"].isin([left, right])].groupby(["task", "responder"]):
        wide = group.pivot_table(index="item_id", columns="style", values="outcome", aggfunc="first").dropna()
        if left not in wide or right not in wide:
            continue
        diff = (wide[right] - wide[left]).to_numpy(float)
        discordant = int(np.sum(wide[right] != wide[left]))
        right_only = int(np.sum((wide[right] == 1) & (wide[left] == 0)))
        p = binomtest(right_only, discordant, 0.5).pvalue if discordant else 1.0
        lo, hi = bootstrap_mean(diff)
        rows.append(
            {"task": task, "responder": responder, "contrast": f"{right}-{left}",
             "n_pairs": len(wide), "left_rate": wide[left].mean(), "right_rate": wide[right].mean(),
             "risk_difference": diff.mean(), "ci_low": lo, "ci_high": hi,
             "discordant": discordant, "right_only": right_only, "p_exact": p}
        )
    return pd.DataFrame(rows)


def adjusted_models(df: pd.DataFrame) -> list[dict]:
    out = []
    for task, group in df[df["style"].isin(["human_style", "llm_style"])].groupby("task"):
        use = group.dropna(subset=["clarity", "politeness"]).copy()
        if use.outcome.nunique() < 2:
            out.append({"task": task, "error": "outcome has no variation"})
            continue
        covariates = []
        for col in ["words", "readability", "clarity", "politeness"]:
            std = use[col].std()
            if std and std > 1e-8:
                use[f"z_{col}"] = (use[col] - use[col].mean()) / std
                covariates.append(f"z_{col}")
        try:
            formula = "outcome ~ C(style, Treatment(reference='human_style')) + C(responder)"
            if covariates:
                formula += " + " + " + ".join(covariates)
            # A linear probability model gives an adjusted risk difference and is
            # stable under the near-separation seen in these small binary cells.
            fit = smf.ols(formula, data=use).fit(cov_type="cluster", cov_kwds={"groups": use["item_id"]})
            term = "C(style, Treatment(reference='human_style'))[T.llm_style]"
            out.append({"task": task, "n": len(use), "adjusted_risk_difference": fit.params[term],
                        "se": fit.bse[term], "p": fit.pvalues[term],
                        "ci_low": fit.conf_int().loc[term, 0], "ci_high": fit.conf_int().loc[term, 1],
                        "included_covariates": covariates})
        except Exception as exc:
            out.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
    return out


def usage_summary() -> dict:
    total_prompt = total_completion = 0
    cost = 0.0
    models: dict[str, int] = {}
    seen: set[str] = set()
    files = list(RAW.glob("*.jsonl")) + list(EVAL.glob("*.jsonl"))
    for path in files:
        for row in read(path):
            for key in ("api", "judge_api"):
                api = row.get(key)
                if not isinstance(api, dict):
                    continue
                response_id = api.get("response_id")
                if response_id and response_id in seen:
                    continue
                if response_id:
                    seen.add(response_id)
                usage = api.get("usage", {}) or {}
                total_prompt += int(usage.get("prompt_tokens", 0) or 0)
                total_completion += int(usage.get("completion_tokens", 0) or 0)
                cost += float(usage.get("cost", 0) or 0)
                model = api.get("model_returned") or api.get("model_requested") or "unknown"
                models[model] = models.get(model, 0) + 1
    return {"unique_api_calls": len(seen), "prompt_tokens": total_prompt, "completion_tokens": total_completion,
            "reported_cost_usd": round(cost, 4), "api_records_by_model": models}


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = flatten_scores(EVAL / "behavior_scores.jsonl")
    # Duplicate cache rows can arise only from retries; retain the last successful key.
    df = df.drop_duplicates(["item_id", "style", "responder"], keep="last")
    df.to_csv(EVAL / "behavior_tidy.csv", index=False)

    validation = pd.DataFrame([{k: x.get(k) for k in ["item_id", "task", "style", "fidelity", "addition", "omission", "clarity", "politeness", "p_ai", "prompt"]} for x in read(EVAL / "rewrite_validation.jsonl") if "error" not in x])
    validation = validation.drop_duplicates(["item_id", "style"], keep="last")
    validation["words"] = validation.prompt.fillna("").str.split().str.len()
    manipulation = validation.groupby(["task", "style"]).agg(n=("p_ai", "size"), p_ai_mean=("p_ai", "mean"), p_ai_sd=("p_ai", "std"), fidelity_mean=("fidelity", "mean"), words_mean=("words", "mean"), clarity_mean=("clarity", "mean"), politeness_mean=("politeness", "mean")).reset_index()
    manipulation.to_csv(EVAL / "manipulation_checks.csv", index=False)

    primary = paired_table(df, "human_style", "llm_style")
    original = paired_table(df, "original", "human_style")
    # Holm correction only for preregistered 3 tasks x 2 responders; benign over-refusal is secondary.
    mask = primary.task.isin(["harmful_compliance", "sycophancy", "accuracy"])
    primary["p_holm"] = np.nan
    primary["reject_holm"] = False
    if mask.sum():
        reject, padj, _, _ = multipletests(primary.loc[mask, "p_exact"], method="holm")
        primary.loc[mask, "p_holm"] = padj
        primary.loc[mask, "reject_holm"] = reject
    primary.to_csv(EVAL / "paired_primary.csv", index=False)
    original.to_csv(EVAL / "paired_paraphrase_baseline.csv", index=False)

    label_df = flatten_scores(EVAL / "label_scores.jsonl")
    if not label_df.empty:
        raw_labels = read(EVAL / "label_scores.jsonl")
        label_map = {(x["item_id"], x["style"], x["responder"], x.get("label")): x for x in raw_labels if "error" not in x and x.get("outcome") is not None}
        label_rows = []
        for (item, style, responder, label), x in label_map.items():
            label_rows.append({"item_id": item, "task": x["task"], "style": style, "responder": responder, "label": label, "outcome": int(x["outcome"])})
        ldf = pd.DataFrame(label_rows)
        lab = []
        for keys, g in ldf.groupby(["task", "style", "responder"]):
            wide = g.pivot(index="item_id", columns="label", values="outcome").dropna()
            if {"AI", "human"}.issubset(wide.columns):
                d = (wide["AI"] - wide["human"]).to_numpy(float); lo, hi = bootstrap_mean(d)
                lab.append(dict(zip(["task", "style", "responder"], keys)) | {"n_pairs": len(wide), "ai_minus_human": d.mean(), "ci_low": lo, "ci_high": hi})
        pd.DataFrame(lab).to_csv(EVAL / "label_effects.csv", index=False)

    adjusted = adjusted_models(df)
    (EVAL / "adjusted_models.json").write_text(json.dumps(adjusted, indent=2))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=manipulation, x="task", y="p_ai_mean", hue="style")
    plt.ylabel("Mean judged probability AI-written (%)"); plt.xlabel(""); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(FIG / "authorship_manipulation.png", dpi=180); plt.close()

    plot = primary.copy(); plot["label"] = plot.task + "\n" + plot.responder.str.split("/").str[-1]
    plt.figure(figsize=(9, 6)); plt.axvline(0, color="black", linewidth=1)
    plt.errorbar(plot.risk_difference, range(len(plot)), xerr=[plot.risk_difference-plot.ci_low, plot.ci_high-plot.risk_difference], fmt="o")
    plt.yticks(range(len(plot)), plot.label); plt.xlabel("LLM-style minus human-style outcome rate")
    plt.tight_layout(); plt.savefig(FIG / "paired_effects.png", dpi=180); plt.close()

    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob("*.jsonl"))}
    summary = {
        "n_scored_unique": len(df), "valid_rate": float(df.valid.mean()),
        "complete_primary_rows": len(primary), "usage": usage_summary(), "raw_sha256": hashes,
        "adjusted_models": adjusted,
    }
    (EVAL / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("MANIPULATION\n", manipulation.to_string(index=False))
    print("\nPRIMARY\n", primary.to_string(index=False))
    print("\nSUMMARY\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
