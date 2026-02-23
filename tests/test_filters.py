from datetime import UTC, datetime

import pytest

from filters import is_potential_startup_paper
from models import Paper


def _paper(title: str, abstract: str = "") -> Paper:
    return Paper(
        paper_id="test.00001",
        title=title,
        url="https://huggingface.co/papers/test.00001",
        abstract=abstract,
        published_at=datetime(2026, 2, 22, tzinfo=UTC),
    )


@pytest.mark.parametrize("title", [
    "A New Benchmark for Language Model Evaluation",
    "Leaderboard Results on Coding Tasks",
    "SemEval 2026 Shared Task: Natural Language Inference",
    "A Large-Scale Dataset for Code Generation",
    "OpenCorpus: A New Corpus for Pretraining",
    "A Survey of Large Language Models",
    "A Review of Reinforcement Learning Methods",
    "Benchmarking Vision-Language Models on Common Tasks",
])
def test_low_value_titles_return_false(title: str) -> None:
    assert is_potential_startup_paper(_paper(title)) is False


@pytest.mark.parametrize("title", [
    "A New Framework for Agentic Task Planning",
    "An End-to-End System for Multimodal Reasoning",
    "LLM Agents for Automated Code Review",
    "Scalable Platform for Real-Time World Models",
    "A Multimodal Architecture for Vision-Language Action",
])
def test_high_value_titles_return_true(title: str) -> None:
    assert is_potential_startup_paper(_paper(title)) is True


def test_ambiguous_title_returns_true() -> None:
    """No clear signal → lean permissive."""
    assert is_potential_startup_paper(_paper("Improving LLM Factuality with References")) is True


def test_benchmark_with_agentic_override_returns_true() -> None:
    """High-value token overrides low-value benchmark signal."""
    assert is_potential_startup_paper(_paper("BenchAgent: Benchmark for Agentic Systems")) is True


def test_dataset_with_pipeline_override_returns_true() -> None:
    """High-value 'pipeline' overrides low-value 'dataset'."""
    assert is_potential_startup_paper(_paper("A Dataset-Driven Pipeline for Medical Triage")) is True


def test_low_value_signal_in_abstract_only() -> None:
    """Low-value token in abstract (not title) is still caught."""
    paper = _paper(
        title="Scaling Neural Networks",
        abstract="We release a large benchmark dataset for evaluation.",
    )
    assert is_potential_startup_paper(paper) is False


def test_high_value_signal_in_abstract_overrides() -> None:
    """High-value token in abstract overrides low-value token in title."""
    paper = _paper(
        title="A Survey of Pretraining Techniques",
        abstract="We propose a new agentic framework built on top of this survey.",
    )
    assert is_potential_startup_paper(paper) is True
