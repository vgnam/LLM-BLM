"""Deterministic, auditable system-prompt diversification for repeated LLM calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from typing import Callable, TypeVar


ENSEMBLE_NAME = "diversified_v1"

_ROLES = (
    "Act as a conservative quantitative forecaster",
    "Act as a skeptical portfolio researcher",
    "Act as a distribution-aware market analyst",
    "Act as a risk-first empirical forecaster",
    "Act as a disciplined cross-sectional analyst",
    "Act as a base-rate-oriented investment researcher",
)

_LENSES = (
    "Balance persistence against mean reversion and avoid overstating weak evidence",
    "Emphasize robustness to outliers and distinguish signal from sampling noise",
    "Consider both central tendency and downside behavior before committing to a view",
    "Use a probabilistic framing and keep uncertainty explicit when the supplied history is ambiguous",
    "Check whether apparent performance differences are economically meaningful rather than merely numerical",
)

T = TypeVar("T")


def diversified_system_prompt(base: str, call_id: int, task_identity: str) -> str:
    """Return a unique prompt while preserving the task's invariant output contract."""

    if call_id < 0:
        raise ValueError("call_id must be non-negative")
    role = _ROLES[(call_id // len(_LENSES)) % len(_ROLES)]
    lens = _LENSES[call_id % len(_LENSES)]
    return (
        f"{role}. {lens}. This is independent ensemble specification "
        f"{call_id + 1} for task {task_identity}; do not assume access to any information "
        "outside the supplied payload.\n\n"
        f"{base}"
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def collect_repeated_calls(
    call_once: Callable[[int], T],
    repeats: int,
    workers: int,
    retry_calls: int,
) -> tuple[list[tuple[int, T]], list[str], int]:
    """Collect successful calls with a unique call ID for every attempt in a task."""

    if repeats < 1 or workers < 1 or retry_calls < 0:
        raise ValueError("repeats/workers must be positive and retry_calls non-negative")
    results: dict[int, T] = {}
    errors: list[str] = []
    attempts = 0
    maximum_attempts = repeats + retry_calls
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while len(results) < repeats and attempts < maximum_attempts:
            batch_size = min(workers, repeats - len(results), maximum_attempts - attempts)
            call_ids = list(range(attempts, attempts + batch_size))
            attempts += batch_size
            if executor:
                futures = {executor.submit(call_once, call_id): call_id for call_id in call_ids}
                for future in as_completed(futures):
                    call_id = futures[future]
                    try:
                        results[call_id] = future.result()
                    except Exception as error:
                        errors.append(f"prompt_call_id={call_id}: {error}")
            else:
                for call_id in call_ids:
                    try:
                        results[call_id] = call_once(call_id)
                    except Exception as error:
                        errors.append(f"prompt_call_id={call_id}: {error}")
    finally:
        if executor:
            executor.shutdown(wait=True)
    ordered = sorted(results.items())[:repeats]
    return ordered, errors, attempts
