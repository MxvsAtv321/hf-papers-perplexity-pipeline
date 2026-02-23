"""Heuristic pre-filter for startup viability (no LLM calls)."""

from __future__ import annotations

from models import Paper

# Tokens that strongly signal low startup value when present without a
# high-value counterpart (pure benchmarks, dataset releases, surveys).
_LOW_VALUE_TOKENS: frozenset[str] = frozenset({
    "benchmark",
    "leaderboard",
    "challenge",
    "shared task",
    "shared-task",
    "dataset",
    "corpus",
    "collection",
    "survey",
    "a survey of",
    "a review of",
    "benchmarking",
})

# Tokens that suggest a productizable system or capability — override low-value
# signals when both are present (e.g. "BenchAgent: Benchmark for Agentic Systems").
_HIGH_VALUE_TOKENS: frozenset[str] = frozenset({
    "framework",
    "system",
    "architecture",
    "platform",
    "pipeline",
    "agents",
    "agentic",
    "world model",
    "multimodal",
    "llm-based",
    "llm agents",
    "end-to-end",
})


def is_potential_startup_paper(paper: Paper) -> bool:
    """Return True if the paper is likely to have startup potential.

    Decision logic (keyword-based, no LLM):
    - False  — low-value tokens present AND no high-value override.
    - True   — high-value tokens present, OR ambiguous (neither signal found).

    Intentionally permissive: prefer False Negatives over False Positives
    so we don't discard promising papers.
    """
    text = f"{paper.title} {paper.abstract or ''}".lower()

    has_low = any(tok in text for tok in _LOW_VALUE_TOKENS)
    has_high = any(tok in text for tok in _HIGH_VALUE_TOKENS)

    if has_low and not has_high:
        return False  # clearly low value with no redeeming signal

    return True  # high-value signal present, or ambiguous → keep
