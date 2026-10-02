"""OpenRouter helpers with retries, structured parsing, and append-only caching."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
LOCK = threading.Lock()


def extract_json(text: str) -> dict:
    """Parse JSON even if a provider wraps it in a markdown fence."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_model(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 500,
    json_mode: bool = False,
    attempts: int = 6,
) -> dict:
    """Call a real model and retain response metadata needed for reproducibility."""
    key = os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_KEY is not set")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            return {
                "text": choice["message"].get("content") or "",
                "model_requested": model,
                "model_returned": body.get("model"),
                "provider": body.get("provider"),
                "usage": body.get("usage", {}),
                "created": body.get("created"),
                "finish_reason": choice.get("finish_reason"),
                "response_id": body.get("id"),
            }
        except Exception as exc:  # network/provider failures are retriable and logged by caller
            last_error = exc
            time.sleep(min(30, 2**attempt + random.random()))
    raise RuntimeError(f"OpenRouter call failed after {attempts} attempts: {last_error}")


def load_cache(path: Path, key_field: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row[key_field])] = row
    return rows


def cached_parallel_map(
    jobs: Iterable[dict],
    worker: Callable[[dict], dict],
    cache_path: Path,
    key_field: str = "job_id",
    max_workers: int = 12,
) -> list[dict]:
    """Run missing jobs concurrently and checkpoint each completed response."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path, key_field)
    missing = [job for job in jobs if str(job[key_field]) not in cache]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, job): job for job in missing}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {**job, "error": f"{type(exc).__name__}: {exc}"}
            with LOCK:
                with cache_path.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache[str(row[key_field])] = row
    return [cache[str(job[key_field])] for job in jobs]
