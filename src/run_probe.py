"""Train an authorship probe, compare it with evaluation awareness, and steer Qwen3-8B."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from api_utils import cached_parallel_map
from run_behavioral import score_worker

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SEED = 20261002
DEVICE = "cuda"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_text(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def extract(model, tokenizer, texts: list[str], batch_size: int = 8) -> np.ndarray:
    all_layers = []
    for start in range(0, len(texts), batch_size):
        formatted = [prompt_text(tokenizer, x) for x in texts[start : start + batch_size]]
        batch = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        out = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
        last = batch["attention_mask"].sum(1) - 1
        layers = torch.stack([h[torch.arange(len(last), device=DEVICE), last] for h in out.hidden_states])
        all_layers.append(layers.float().cpu().numpy().transpose(1, 0, 2))
    return np.concatenate(all_layers, axis=0)


def build_authorship() -> tuple[list[str], np.ndarray, np.ndarray, list[dict]]:
    rewrites = {x["item_id"]: x for x in read(ROOT / "results/model_outputs/rewrites.jsonl") if "error" not in x and x.get("human_style") and x.get("llm_style")}
    vals = {x["job_id"]: x for x in read(ROOT / "results/evaluations/rewrite_validation.jsonl") if "error" not in x}
    texts, labels, groups, meta = [], [], [], []
    for item, row in rewrites.items():
        hv, lv = vals.get(f"{item}::human_style", {}), vals.get(f"{item}::llm_style", {})
        if not all(int(v.get("fidelity", 0)) >= 4 and not v.get("addition", True) and not v.get("omission", True) for v in (hv, lv)):
            continue
        for style, label in (("human_style", 0), ("llm_style", 1)):
            texts.append(row[style]); labels.append(label); groups.append(item)
            meta.append({"item_id": item, "task": row["task"], "style": style, "text": row[style]})
    return texts, np.asarray(labels), np.asarray(groups), meta


def split_groups(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dummy = np.zeros(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval, test = next(outer.split(dummy, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    tr_rel, val_rel = next(inner.split(dummy[trainval], groups=groups[trainval]))
    return trainval[tr_rel], trainval[val_rel], test


def parse_eval_pairs(n: int = 80) -> tuple[list[str], np.ndarray]:
    data = read_json(ROOT / "code/evaluation-awareness-probing/datasets/tom_vs_sharegpt.json")
    texts, labels = [], []
    for row in data[:n]:
        q = row["question"]
        parts = q.split("\n\n(B)", 1)
        if len(parts) != 2 or "(A)" not in parts[0]:
            continue
        a = parts[0].split("(A)", 1)[1].strip(); b = parts[1].strip()
        pos = row["positive"].strip()
        eval_text, deploy_text = (a, b) if pos == "(A)" else (b, a)
        texts.extend([eval_text, deploy_text]); labels.extend([1, 0])
    return texts, np.asarray(labels)


def read_json(path: Path):
    return json.loads(path.read_text())


@torch.inference_mode()
def generate_condition(model, tokenizer, rows: list[dict], layer: int, vector: torch.Tensor | None, name: str) -> list[dict]:
    outputs = []
    handle = None
    if vector is not None:
        vec = vector.to(device=DEVICE, dtype=model.dtype)
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + vec.view(1, 1, -1),) + output[1:]
            return output + vec.view(1, 1, -1)
        handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for start in range(0, len(rows), 8):
            chunk = rows[start : start + 8]
            strings = [prompt_text(tokenizer, x["prompt"]) for x in chunk]
            batch = tokenizer(strings, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
            generated = model.generate(**batch, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            for i, row in enumerate(chunk):
                new = generated[i, batch["input_ids"].shape[1] :]
                outputs.append({**row, "steering_condition": name, "response": tokenizer.decode(new, skip_special_tokens=True).strip(), "responder": MODEL_ID, "job_id": f"steer::{row['item_id']}::{name}"})
    finally:
        if handle is not None:
            handle.remove()
    return outputs


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16).to(DEVICE).eval()

    texts, y, groups, meta = build_authorship()
    acts = extract(model, tokenizer, texts)
    train, val, test = split_groups(groups)
    candidate_layers = list(range(4, acts.shape[1], 4)) + [acts.shape[1] - 1]
    records, fits = [], {}
    for layer in sorted(set(candidate_layers)):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
        pipe.fit(acts[train, layer], y[train]); fits[layer] = pipe
        records.append({"layer": layer, "val_auc": roc_auc_score(y[val], pipe.predict_proba(acts[val, layer])[:, 1])})
    best = max(records, key=lambda x: x["val_auc"])["layer"]
    clf = fits[best]
    test_prob = clf.predict_proba(acts[test, best])[:, 1]
    test_auc = roc_auc_score(y[test], test_prob)
    test_bal = balanced_accuracy_score(y[test], test_prob >= 0.5)

    # A surface-only comparator on the identical split.
    surf = np.array([[len(x.split()), len(x), x.count("\n")] for x in texts], float)
    surface_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)).fit(surf[train], y[train])
    surface_auc = roc_auc_score(y[test], surface_clf.predict_proba(surf[test])[:, 1])

    human_mean = acts[train[y[train] == 0], best].mean(0)
    machine_mean = acts[train[y[train] == 1], best].mean(0)
    direction = machine_mean - human_mean
    direction /= np.linalg.norm(direction)
    proj_sd = float(np.std(acts[train, best] @ direction))
    scaler = clf.named_steps["standardscaler"]
    logistic = clf.named_steps["logisticregression"]
    effective_coef = logistic.coef_[0] / scaler.scale_
    author_logit_shift = float(effective_coef @ (direction * proj_sd))
    shift_rng = np.random.default_rng(SEED)
    random_logit_shifts = []
    for _ in range(5):
        random_vec = shift_rng.normal(size=direction.shape); random_vec /= np.linalg.norm(random_vec)
        random_logit_shifts.append(float(effective_coef @ (random_vec * proj_sd)))

    eval_texts, eval_y = parse_eval_pairs()
    eval_acts = extract(model, tokenizer, eval_texts)[:, best]
    eval_dir = eval_acts[eval_y == 1].mean(0) - eval_acts[eval_y == 0].mean(0)
    eval_dir /= np.linalg.norm(eval_dir)
    cosine = float(direction @ eval_dir)

    result = {
        "model": MODEL_ID, "revision": REVISION, "n_authorship_texts": len(texts),
        "n_items": len(set(groups)), "train_n": len(train), "validation_n": len(val), "test_n": len(test),
        "selected_layer": best, "validation_layer_sweep": records, "test_auc": test_auc,
        "test_balanced_accuracy": test_bal, "surface_only_test_auc": surface_auc,
        "projection_sd": proj_sd, "authorship_eval_direction_cosine": cosine,
        "expected_probe_logit_shift_author_pos": author_logit_shift,
        "expected_probe_logit_shift_author_neg": -author_logit_shift,
        "expected_probe_logit_shifts_random": random_logit_shifts,
        "gpu": torch.cuda.get_device_name(0), "dtype": str(model.dtype),
    }
    (ROOT / "results/evaluations/probe_results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(ROOT / "results/evaluations/probe_artifacts.npz", direction=direction, eval_direction=eval_dir, effective_probe_coef=effective_coef, test_y=y[test], test_prob=test_prob, test_groups=groups[test])

    if test_auc >= 0.70:
        # Decoder-only batched generation requires left padding; activation extraction
        # above intentionally used right padding so the final non-padding index was simple.
        tokenizer.padding_side = "left"
        base = read(ROOT / "results/data/base_items.jsonl")
        rew = {x["item_id"]: x for x in read(ROOT / "results/model_outputs/rewrites.jsonl") if "error" not in x}
        selected = []
        for task in sorted({x["task"] for x in base}):
            candidates = [x for x in base if x["task"] == task and x["item_id"] in rew and rew[x["item_id"]].get("human_style")]
            for row in candidates[:4]:
                selected.append({**row, "style": "human_style", "prompt": rew[row["item_id"]]["human_style"]})
        rng = np.random.default_rng(SEED)
        vectors = {"none": None, "author_pos": torch.tensor(direction * proj_sd), "author_neg": torch.tensor(-direction * proj_sd)}
        for i in range(5):
            vec = rng.normal(size=direction.shape); vec /= np.linalg.norm(vec)
            vectors[f"random_{i}"] = torch.tensor(vec * proj_sd)
        out_path = ROOT / "results/model_outputs/steering_responses.jsonl"
        if out_path.exists():
            generations = read(out_path)
        else:
            generations = []
            for name, vector in vectors.items():
                generations.extend(generate_condition(model, tokenizer, selected, best - 1, vector, name))
            out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in generations))
        scores = cached_parallel_map(generations, score_worker, ROOT / "results/evaluations/steering_scores.jsonl", max_workers=12)
        result["steering_jobs"] = len(generations); result["steering_scored"] = sum("error" not in x for x in scores)
        result["runtime_seconds"] = time.time() - started
        (ROOT / "results/evaluations/probe_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
