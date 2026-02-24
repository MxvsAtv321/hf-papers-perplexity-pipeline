"""Pattern recognition and composite idea generation over pipeline CSV outputs.

This module operates as a *meta-layer* on top of existing pipeline CSVs.
It never calls the HF API or LLM APIs directly — it builds prompts that
mine_patterns.py (the CLI) can forward to an LLM.

Public API
----------
load_papers(csv_paths, min_score) -> list[AnalyzedPaper]
extract_themes(papers)            -> list[Theme]
find_composite_candidates(theme, papers_by_id) -> list[list[AnalyzedPaper]]
build_theme_summary_prompt(theme, papers)       -> list[dict]   # messages
build_composite_prompt(theme, candidates)       -> list[dict]   # messages
themes_to_rows(themes)            -> list[dict]  # for CSV writing
composite_ideas_to_rows(ideas)    -> list[dict]  # for CSV writing

load_composite_ideas(csv_path)               -> list[CompositeIdea]
aggregate_paper_signals(paper_ids, papers)   -> dict
build_scoring_prompt(idea, signals)          -> list[dict]   # messages
parse_score_response(raw)                    -> dict | None
score_composite_ideas(ideas, papers, llm_fn) -> list[ScoredCompositeIdea]
scored_ideas_to_rows(ideas)                  -> list[dict]   # for CSV writing
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import os
import re
import uuid
from typing import Any, Callable, NamedTuple

LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzedPaper(NamedTuple):
    """Normalized view of a pipeline CSV row, schema-agnostic."""
    paper_id: str
    title: str
    url: str
    abstract: str
    startup_potential: int | None
    market_pull: int | None
    technical_moat: int | None
    story_for_accelerator: int | None
    overall_score: int | None
    claude_final_score: int | None
    claude_final_label: str | None  # keep | maybe | drop | unknown | None
    capability: str | None          # plain-language capability (Stage 4 only)
    product_angles_text: str | None # collapsed product angle names
    score_rationale: str | None     # Stage 2 OpenAI rationale


class Theme(NamedTuple):
    name: str
    keywords: list[str]
    paper_ids: list[str]
    summary: str | None   # filled in by LLM after prompt; None until then


class CompositeIdea(NamedTuple):
    id: str
    title: str
    theme_name: str
    paper_ids: list[str]
    core_capability: str
    added_value: str
    target_user: str
    problem_statement: str
    wedge_description: str
    risks: str


# ─────────────────────────────────────────────────────────────────────────────
# Theme seed keywords  (order matters: first theme with highest overlap wins)
# ─────────────────────────────────────────────────────────────────────────────

THEME_SEEDS: dict[str, list[str]] = {
    "Web & Computer-Use Agents": [
        "web agent", "browser", "computer use", "computer-use", "desktop",
        "rpa", "gui agent", "ui automation", "web automation", "web navigation",
        "computer using", "webpage", "webworld",
    ],
    "Agent Infrastructure & Orchestration": [
        "orchestrat", "multi-agent", "agent framework", "agent router",
        "tool use", "skill transfer", "routing", "skill orchestra",
        "agent system", "compound ai", "agent infrast",
    ],
    "Robotics & Embodied AI": [
        "robot", "embodied", "manipulation", "drone", "grasp",
        "locomotion", "dexterous", "sim-to-real", "pick-and-place",
        "robocurate", "frappe", "generalist polic",
    ],
    "3D Reconstruction & Spatial AI": [
        "3d reconstruct", "gaussian splat", "nerf", "scene reconstruct",
        "novel view", "point cloud", "mesh reconstruct", "tttlrm",
        "reality capture", "3d model",
    ],
    "World Models & Simulation": [
        "world model", "world simulat", "4d world", "environment model",
        "predictive model", "generated reality", "interactive video",
        "cuwm", "computer-using world",
    ],
    "Video Generation & Editing": [
        "video generat", "video diffus", "text-to-video", "video synthesis",
        "video editing", "video model", "temporal video", "ddit",
    ],
    "On-Device & Edge AI": [
        "on-device", "mobile", "edge deploy", "model compress",
        "quantiz", "on device", "mobile-o", "device inference",
    ],
    "Inference & Training Efficiency": [
        "inference optim", "kernel", "throughput", "latency reduction",
        "training efficiency", "flops", "compute efficient", "k-search",
        "ddit", "unified latent", "accelerat",
    ],
    "Reward Modeling & RL": [
        "reward model", "reinforcement learn", "rlhf", "preference optim",
        "dpo", "ppo", "policy gradient", "topreward", "reward signal",
    ],
    "Multimodal Models": [
        "multimodal", "vision-language", "vlm", "vision language",
        "image-text", "visual question", "vqa", "image generation",
        "understand and generat",
    ],
    "Video Understanding & Segmentation": [
        "video segmentat", "video understand", "video tracking",
        "object tracking", "instance segmentat", "videomask", "vieomt",
        "video reasoning",
    ],
    "AI Safety & Red-Teaming": [
        "safety", "red-team", "jailbreak", "adversarial", "guard",
        "harmful", "robustness", "attack", "agents of chaos",
    ],
    "Time-Series & Industrial AI": [
        "time-series", "time series", "temporal forecast", "anomaly detect",
        "sensor", "industrial", "iot", "sentsrbench", "root cause",
    ],
    "Causal & Reasoning AI": [
        "causal discover", "causal model", "causal graph", "reasoning",
        "chain-of-thought", "logical", "temporal causal",
    ],
    "Synthetic Data & Data Engines": [
        "synthetic data", "data engine", "data curat", "data generat",
        "augment", "action-verified", "trajectory", "robocurate",
    ],
    "Code & Software Engineering Agents": [
        "code generat", "software engineer", "swe", "programming agent",
        "debug", "repo", "github", "patch generation", "swe-universe",
    ],
    "Biology & Life Sciences AI": [
        "protein", "drug discover", "molecular", "genomic", "aav",
        "biolog", "therapeut", "gene edit", "aavgen",
    ],
    "Digital Humans & Avatars": [
        "avatar", "digital human", "motion generat", "gesture", "talking head",
        "body motion", "animation", "sarah", "spatially aware",
    ],
    "XR & Spatial Computing": [
        "xr", "augmented reality", "virtual reality", "mixed reality",
        "spatial comput", "hologram", "vr headset", "ar application",
        "generated reality", "egocentric",
    ],
    "LLM Architecture & Foundation Models": [
        "pretrain", "language model", "transformer", "attention mechanism",
        "mixture of experts", "moe", "scaling law", "foundation model",
        "llm architect", "muon", "optimizer",
    ],
    "Evaluation & Benchmarking": [
        "benchmark", "evaluation suite", "leaderboard", "metric",
        "test suite", "human eval", "assessment", "swe-universe", "webworld-bench",
    ],
    "Human-AI Interaction": [
        "human-ai", "human interaction", "user study", "intervention",
        "adaptive autonomy", "conversational", "human-in-the-loop",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — handles both pipeline CSVs and report CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_papers(
    csv_paths: list[str],
    min_score: int = 1,
) -> list[AnalyzedPaper]:
    """Load and deduplicate papers from one or more pipeline/report CSVs.

    Handles both the full pipeline schema (papers_wide_scout.csv,
    papers_pipeline.csv) and the report schema (papers_debated.csv,
    papers_top_picks.csv).  Papers with overall_score < min_score are dropped.
    Deduplication is by paper_id; first occurrence wins.

    Args:
        csv_paths: Paths to one or more CSV files to load.
        min_score: Minimum overall_score (1–5) to keep a paper. Default 1.

    Returns:
        Deduplicated, score-filtered list of AnalyzedPaper objects.
    """
    seen: dict[str, AnalyzedPaper] = {}
    total_read = 0

    for path in csv_paths:
        if not os.path.exists(path):
            LOGGER.warning("CSV not found, skipping: %s", path)
            continue

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        LOGGER.info("Loaded %s rows from %s", len(rows), path)
        total_read += len(rows)

        for row in rows:
            paper = _row_to_paper(row)
            if paper is None:
                continue
            if (paper.overall_score or 0) < min_score:
                continue
            if paper.paper_id not in seen:
                seen[paper.paper_id] = paper

    LOGGER.info(
        "load_papers: read=%s unique=%s after_min_score=%s",
        total_read, len(seen), len(seen),
    )
    return list(seen.values())


def _row_to_paper(row: dict[str, str]) -> AnalyzedPaper | None:
    """Normalize a raw CSV row into an AnalyzedPaper, tolerating both schemas."""
    # paper_id: direct field or extracted from URL
    paper_id = _str(row.get("paper_id"))
    if not paper_id:
        url = _str(row.get("url")) or ""
        paper_id = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    if not paper_id:
        return None

    title = _str(row.get("title"))
    if not title:
        return None

    url = _str(row.get("url")) or f"https://huggingface.co/papers/{paper_id}"

    # overall_score: pipeline schema uses "overall_score"; report schema "openai_overall"
    overall_score = _int(row.get("overall_score") or row.get("openai_overall"))

    # startup_potential / market_pull / etc.
    startup_potential = _int(
        row.get("startup_potential") or row.get("openai_startup")
    )
    market_pull = _int(row.get("market_pull") or row.get("openai_market"))
    technical_moat = _int(row.get("technical_moat") or row.get("openai_moat"))
    story = _int(
        row.get("story_for_accelerator") or row.get("openai_story")
    )

    # Claude scores: pipeline schema uses full names; report schema uses "final_score"
    claude_final_score = _int(
        row.get("claude_final_score") or row.get("final_score")
    )
    claude_final_label = _str(
        row.get("claude_final_label") or row.get("final_label")
    )

    # Capability: pipeline uses long name; report uses "capability"
    capability = _str(
        row.get("capability_plain_language_capability") or row.get("capability")
    )

    # Product angles: collapse JSON array of objects to a string of angle names
    product_angles_text = _collapse_product_angles(
        row.get("product_angles") or row.get("best_product_angle") or ""
    )

    score_rationale = _str(row.get("score_rationale") or row.get("rationale"))
    abstract = _str(row.get("abstract")) or ""

    return AnalyzedPaper(
        paper_id=paper_id,
        title=title,
        url=url,
        abstract=abstract,
        startup_potential=startup_potential,
        market_pull=market_pull,
        technical_moat=technical_moat,
        story_for_accelerator=story,
        overall_score=overall_score,
        claude_final_score=claude_final_score,
        claude_final_label=claude_final_label,
        capability=capability,
        product_angles_text=product_angles_text,
        score_rationale=score_rationale,
    )


def _collapse_product_angles(raw: str) -> str | None:
    """Turn a JSON product_angles array into a comma-joined string of names."""
    if not raw or raw == "[]":
        return None
    try:
        angles = json.loads(raw)
        if isinstance(angles, list):
            names = [a.get("name", "") for a in angles if isinstance(a, dict)]
            return ", ".join(n for n in names if n) or None
        # Report CSVs store best_product_angle as a plain string
        if isinstance(angles, str):
            return angles or None
    except (json.JSONDecodeError, AttributeError):
        # It's already a plain string (report CSV)
        return raw.strip() or None
    return None


def _str(value: Any) -> str | None:
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Theme extraction — keyword-overlap scoring
# ─────────────────────────────────────────────────────────────────────────────

def extract_themes(papers: list[AnalyzedPaper]) -> list[Theme]:
    """Assign each paper to the best-matching theme via keyword overlap.

    Papers with no match (score 0 across all themes) go into "Uncategorized".
    Returns themes sorted by number of members descending; "Uncategorized" last.

    Args:
        papers: List of normalized papers.

    Returns:
        List of Theme objects (only themes with ≥1 member are returned).
    """
    # Assign paper → theme
    theme_to_ids: dict[str, list[str]] = {name: [] for name in THEME_SEEDS}
    theme_to_ids["Uncategorized"] = []

    for paper in papers:
        text = _build_search_text(paper)
        best_theme = "Uncategorized"
        best_score = 0

        for theme_name, keywords in THEME_SEEDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_theme = theme_name

        theme_to_ids[best_theme].append(paper.paper_id)

    # Build Theme objects, drop empty themes
    themes: list[Theme] = []
    for name, paper_ids in theme_to_ids.items():
        if not paper_ids:
            continue
        keywords = list(THEME_SEEDS.get(name, []))
        themes.append(Theme(name=name, keywords=keywords, paper_ids=paper_ids, summary=None))

    # Sort: most papers first, Uncategorized last
    themes.sort(
        key=lambda t: (t.name == "Uncategorized", -len(t.paper_ids))
    )

    LOGGER.info(
        "extract_themes: %s themes from %s papers",
        len(themes),
        len(papers),
    )
    return themes


def _build_search_text(paper: AnalyzedPaper) -> str:
    """Concatenate all text fields into a single lower-cased search blob."""
    parts = [
        paper.title or "",
        (paper.abstract or "")[:400],
        paper.capability or "",
        paper.score_rationale or "",
        paper.product_angles_text or "",
    ]
    return " ".join(parts).lower()


def score_paper_against_theme(paper: AnalyzedPaper, keywords: list[str]) -> int:
    """Return the number of theme keywords that appear in the paper's text."""
    text = _build_search_text(paper)
    return sum(1 for kw in keywords if kw in text)


# ─────────────────────────────────────────────────────────────────────────────
# Composite candidate identification
# ─────────────────────────────────────────────────────────────────────────────

_MIN_COMPOSITE_SCORE = 3  # both papers must have overall_score >= this
_MAX_CANDIDATES_PER_THEME = 3


def find_composite_candidates(
    theme: Theme,
    papers_by_id: dict[str, AnalyzedPaper],
) -> list[list[AnalyzedPaper]]:
    """Find pairs (or triples) of papers within a theme that are complementary.

    Complementarity heuristics (ANY one is sufficient for a pair):
    1. One paper has technical_moat >= 4 AND the other has market_pull >= 4
       (strong tech + strong market signal).
    2. One paper has deep analysis (capability not None) and the other is
       a shallow paper with startup_potential >= 4 (capability + market pull combo).
    3. One paper has high story_for_accelerator (>=4) and the other has high
       technical_moat (>=4) but lower story score (narrative + moat combo).

    Only papers with overall_score >= _MIN_COMPOSITE_SCORE are considered.
    Returns at most _MAX_CANDIDATES_PER_THEME candidate pairs.

    Args:
        theme: The theme whose members to search.
        papers_by_id: Full paper lookup dict.

    Returns:
        List of [paper1, paper2] lists (or [p1, p2, p3] when a strong triple found).
    """
    members = [
        papers_by_id[pid]
        for pid in theme.paper_ids
        if pid in papers_by_id
    ]
    eligible = [p for p in members if (p.overall_score or 0) >= _MIN_COMPOSITE_SCORE]

    candidates: list[list[AnalyzedPaper]] = []

    for p1, p2 in itertools.combinations(eligible, 2):
        if _are_complementary(p1, p2):
            candidates.append([p1, p2])
            if len(candidates) >= _MAX_CANDIDATES_PER_THEME:
                break

    return candidates


def _are_complementary(p1: AnalyzedPaper, p2: AnalyzedPaper) -> bool:
    """Return True if the two papers complement each other across key dimensions."""
    moat1 = p1.technical_moat or 0
    moat2 = p2.technical_moat or 0
    pull1 = p1.market_pull or 0
    pull2 = p2.market_pull or 0
    story1 = p1.story_for_accelerator or 0
    story2 = p2.story_for_accelerator or 0

    # Heuristic 1: tech-moat on one side, market-pull on the other
    if (moat1 >= 4 and pull2 >= 4) or (moat2 >= 4 and pull1 >= 4):
        return True

    # Heuristic 2: deep-analysis paper paired with high-startup-potential shallow paper
    has_capability1 = p1.capability is not None
    has_capability2 = p2.capability is not None
    sp1 = p1.startup_potential or 0
    sp2 = p2.startup_potential or 0
    if (has_capability1 and not has_capability2 and sp2 >= 4) or (
        has_capability2 and not has_capability1 and sp1 >= 4
    ):
        return True

    # Heuristic 3: strong accelerator story + strong technical moat (but not already captured)
    if (story1 >= 4 and moat2 >= 4) or (story2 >= 4 and moat1 >= 4):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — return messages lists, never call APIs
# ─────────────────────────────────────────────────────────────────────────────

_THEME_SUMMARY_SYSTEM = """\
You are a deep-tech VC analyst building a market map of emerging AI research.
Given a cluster of related papers, produce a concise JSON analysis.
Respond ONLY with valid JSON. No prose outside the JSON.

Required schema:
{
  "description": "<2-3 sentence overview of what this capability cluster is about>",
  "startup_opportunities": [
    "<1-line opportunity 1>",
    "<1-line opportunity 2>",
    "<1-line opportunity 3>"
  ],
  "gaps": "<1-2 sentences on what's missing or underdeveloped across the papers in this theme>"
}"""


def build_theme_summary_prompt(
    theme: Theme,
    papers: list[AnalyzedPaper],
) -> list[dict]:
    """Build an OpenAI-style messages list to summarize a theme.

    Args:
        theme: The theme to summarize.
        papers: All papers belonging to this theme.

    Returns:
        messages list suitable for passing directly to an OpenAI chat call.
    """
    lines = [f"Theme: {theme.name}", ""]
    for p in papers:
        score = p.overall_score or p.claude_final_score or "?"
        cap = (p.capability or p.score_rationale or "")[:120]
        lines.append(f"- [{score}] {p.title}")
        if cap:
            lines.append(f"  {cap}")
    user_content = "\n".join(lines)

    return [
        {"role": "system", "content": _THEME_SUMMARY_SYSTEM},
        {"role": "user", "content": user_content},
    ]


_COMPOSITE_SYSTEM = """\
You are a deep-tech startup strategist. Given 2-3 complementary ML papers from the same theme,
decide whether combining their capabilities creates a meaningfully stronger startup concept.

If YES, output a CompositeIdea JSON object.
If NO, output {"combine": false, "reason": "<brief explanation>"}.

CompositeIdea schema:
{
  "combine": true,
  "title": "<human-readable concept name, ≤10 words>",
  "core_capability": "<what the combined system can do that neither paper alone can>",
  "added_value": "<concrete advantage of combining vs. each paper alone>",
  "target_user": "<specific persona in workflow language>",
  "problem_statement": "<the concrete workflow problem this solves>",
  "wedge_description": "<why this is a strong startup wedge — narrow entry point, clear ROI>",
  "risks": "<copy risk, technical unknowns, adoption challenges — 1-2 sentences>"
}

Respond ONLY with valid JSON. Do not force a combination if it does not clearly add value."""


def build_composite_prompt(
    theme: Theme,
    candidates: list[AnalyzedPaper],
) -> list[dict]:
    """Build a messages list to evaluate a composite startup concept.

    Args:
        theme: The theme these candidates belong to.
        candidates: 2–3 complementary papers to consider combining.

    Returns:
        messages list suitable for passing directly to an OpenAI or Claude chat call.
    """
    lines = [f"Theme: {theme.name}", ""]
    for p in candidates:
        lines.append(f"Paper: {p.title}")
        lines.append(f"URL: {p.url}")
        if p.capability:
            lines.append(f"Capability: {p.capability[:200]}")
        if p.score_rationale:
            lines.append(f"Rationale: {p.score_rationale[:150]}")
        if p.product_angles_text:
            lines.append(f"Product angles: {p.product_angles_text[:150]}")
        scores = (
            f"overall={p.overall_score} startup={p.startup_potential} "
            f"market={p.market_pull} moat={p.technical_moat}"
        )
        lines.append(f"Scores: {scores}")
        lines.append("")

    lines.append(
        "Decide: does combining these papers create a stronger startup concept? "
        "Reply with the JSON schema above."
    )

    return [
        {"role": "system", "content": _COMPOSITE_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CSV serialization helpers
# ─────────────────────────────────────────────────────────────────────────────

THEMES_CSV_COLUMNS = [
    "theme_name",
    "num_papers",
    "keywords",
    "paper_ids",
    "summary",
]

COMPOSITES_CSV_COLUMNS = [
    "id",
    "theme_name",
    "paper_ids",
    "title",
    "core_capability",
    "added_value",
    "target_user",
    "problem_statement",
    "wedge_description",
    "risks",
]


def themes_to_rows(themes: list[Theme]) -> list[dict]:
    """Serialize a list of Themes to flat dicts for CSV writing."""
    rows = []
    for t in themes:
        rows.append({
            "theme_name": t.name,
            "num_papers": len(t.paper_ids),
            "keywords": ", ".join(t.keywords[:10]),
            "paper_ids": json.dumps(t.paper_ids),
            "summary": t.summary or "",
        })
    return rows


def composite_ideas_to_rows(ideas: list[CompositeIdea]) -> list[dict]:
    """Serialize a list of CompositeIdeas to flat dicts for CSV writing."""
    rows = []
    for idea in ideas:
        rows.append({
            "id": idea.id,
            "theme_name": idea.theme_name,
            "paper_ids": json.dumps(idea.paper_ids),
            "title": idea.title,
            "core_capability": idea.core_capability,
            "added_value": idea.added_value,
            "target_user": idea.target_user,
            "problem_statement": idea.problem_statement,
            "wedge_description": idea.wedge_description,
            "risks": idea.risks,
        })
    return rows


def write_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    """Write rows to a CSV file, creating or overwriting it."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %s rows to %s", len(rows), path)


def parse_composite_response(
    raw: str,
    theme_name: str,
    paper_ids: list[str],
) -> CompositeIdea | None:
    """Parse an LLM composite-idea response into a CompositeIdea or None.

    Returns None when the LLM decided not to combine ("combine": false).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON from noisy output
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            LOGGER.warning("Could not parse composite LLM response")
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            LOGGER.warning("Could not parse composite LLM response after extraction")
            return None

    if not data.get("combine", True) is True:
        LOGGER.info("LLM decided not to combine: %s", data.get("reason", ""))
        return None

    return CompositeIdea(
        id=str(uuid.uuid4())[:8],
        title=str(data.get("title", "Untitled")),
        theme_name=theme_name,
        paper_ids=paper_ids,
        core_capability=str(data.get("core_capability", "")),
        added_value=str(data.get("added_value", "")),
        target_user=str(data.get("target_user", "")),
        problem_statement=str(data.get("problem_statement", "")),
        wedge_description=str(data.get("wedge_description", "")),
        risks=str(data.get("risks", "")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scoring layer — evaluate composite ideas on 5 axes via LLM
# ─────────────────────────────────────────────────────────────────────────────

class ScoredCompositeIdea(NamedTuple):
    """CompositeIdea enriched with 5-axis LLM scores and a weighted composite_score."""
    id: str
    title: str
    theme_name: str
    paper_ids: list[str]
    core_capability: str
    added_value: str
    target_user: str
    problem_statement: str
    wedge_description: str
    risks: str
    wedge_clarity: int | None       # 1–5
    technical_moat: int | None      # 1–5
    market_pull: int | None         # 1–5
    founder_fit: int | None         # 1–5
    composite_synergy: int | None   # 1–5
    composite_score: float | None   # weighted avg (see parse_score_response)
    scoring_notes: str | None


SCORED_COMPOSITES_CSV_COLUMNS = [
    "id",
    "theme_name",
    "paper_ids",
    "title",
    "core_capability",
    "added_value",
    "target_user",
    "problem_statement",
    "wedge_description",
    "risks",
    "wedge_clarity",
    "technical_moat",
    "market_pull",
    "founder_fit",
    "composite_synergy",
    "composite_score",
    "scoring_notes",
]


def load_composite_ideas(csv_path: str) -> list[CompositeIdea]:
    """Load composite ideas from a previously written CSV file.

    Args:
        csv_path: Path to the composite ideas CSV (papers_composite_ideas.csv).

    Returns:
        List of CompositeIdea objects, or [] if the file doesn't exist.
    """
    if not os.path.exists(csv_path):
        LOGGER.warning("Composite ideas file not found: %s", csv_path)
        return []
    ideas: list[CompositeIdea] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                paper_ids = json.loads(row.get("paper_ids", "[]"))
            except json.JSONDecodeError:
                paper_ids = []
            ideas.append(CompositeIdea(
                id=row.get("id", ""),
                title=row.get("title", ""),
                theme_name=row.get("theme_name", ""),
                paper_ids=paper_ids,
                core_capability=row.get("core_capability", ""),
                added_value=row.get("added_value", ""),
                target_user=row.get("target_user", ""),
                problem_statement=row.get("problem_statement", ""),
                wedge_description=row.get("wedge_description", ""),
                risks=row.get("risks", ""),
            ))
    LOGGER.info("Loaded %s composite ideas from %s", len(ideas), csv_path)
    return ideas


def aggregate_paper_signals(
    paper_ids: list[str],
    papers_by_id: dict[str, "AnalyzedPaper"],
) -> dict[str, Any]:
    """Compute average per-paper signals for a composite idea's source papers.

    Args:
        paper_ids: IDs of the papers that make up the composite idea.
        papers_by_id: Full paper lookup dict.

    Returns:
        Dict with avg_* signal keys plus has_claude and num_papers.
    """
    papers = [papers_by_id[pid] for pid in paper_ids if pid in papers_by_id]
    if not papers:
        return {
            "avg_overall": None,
            "avg_startup_potential": None,
            "avg_market_pull": None,
            "avg_technical_moat": None,
            "avg_story": None,
            "avg_claude_score": None,
            "has_claude": False,
            "num_papers": 0,
        }

    def _avg(vals: list[int | None]) -> float | None:
        valid = [v for v in vals if v is not None]
        return round(sum(valid) / len(valid), 2) if valid else None

    return {
        "avg_overall": _avg([p.overall_score for p in papers]),
        "avg_startup_potential": _avg([p.startup_potential for p in papers]),
        "avg_market_pull": _avg([p.market_pull for p in papers]),
        "avg_technical_moat": _avg([p.technical_moat for p in papers]),
        "avg_story": _avg([p.story_for_accelerator for p in papers]),
        "avg_claude_score": _avg([p.claude_final_score for p in papers]),
        "has_claude": any(p.claude_final_score is not None for p in papers),
        "num_papers": len(papers),
    }


_SCORING_SYSTEM = """\
You are a deep-tech startup advisor scoring a composite startup idea derived from ML research papers.
Score the idea on 5 axes (each 1–5, integer only) and add brief scoring_notes.

Scoring rubric:
- wedge_clarity (1-5): How narrow and concrete is the entry wedge? 5 = crisp wedge, clear first customer, obvious ROI.
- technical_moat (1-5): How defensible is the underlying technology? 5 = novel combination, hard to replicate.
- market_pull (1-5): How strong is market demand? 5 = clear pain, large market, buyers exist today.
- founder_fit (1-5): How likely is a typical ML/AI founding team to execute this? 5 = obvious fit, familiar domain.
- composite_synergy (1-5): Does combining the papers create more value than either alone? 5 = strong synergy, not just concatenation.

Respond ONLY with valid JSON matching this schema:
{
  "wedge_clarity": <int 1-5>,
  "technical_moat": <int 1-5>,
  "market_pull": <int 1-5>,
  "founder_fit": <int 1-5>,
  "composite_synergy": <int 1-5>,
  "scoring_notes": "<2-3 sentences on key strengths and concerns>"
}"""


def build_scoring_prompt(
    idea: "CompositeIdea",
    signals: dict[str, Any],
) -> list[dict]:
    """Build a messages list to score a composite startup idea.

    Args:
        idea: The composite idea to score.
        signals: Aggregated paper signals from aggregate_paper_signals().

    Returns:
        messages list suitable for passing to an OpenAI chat call.
    """
    has_claude = signals.get("has_claude", False)
    avg_claude = signals.get("avg_claude_score")
    claude_note = (
        f"Claude debate score (avg): {avg_claude}"
        if has_claude and avg_claude is not None
        else "No Claude debate data available (Stage 2 signals only)."
    )

    lines = [
        f"Title: {idea.title}",
        f"Theme: {idea.theme_name}",
        f"Papers: {', '.join(idea.paper_ids)}",
        "",
        f"Core capability: {idea.core_capability}",
        f"Added value: {idea.added_value}",
        f"Target user: {idea.target_user}",
        f"Problem statement: {idea.problem_statement}",
        f"Wedge description: {idea.wedge_description}",
        f"Risks: {idea.risks}",
        "",
        "Signals from source papers:",
        f"  avg overall_score:        {signals.get('avg_overall', 'N/A')}",
        f"  avg startup_potential:    {signals.get('avg_startup_potential', 'N/A')}",
        f"  avg market_pull:          {signals.get('avg_market_pull', 'N/A')}",
        f"  avg technical_moat:       {signals.get('avg_technical_moat', 'N/A')}",
        f"  avg story_for_accel:      {signals.get('avg_story', 'N/A')}",
        f"  {claude_note}",
        "",
        "Score this composite idea using the rubric in your system prompt.",
    ]

    return [
        {"role": "system", "content": _SCORING_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_score_response(raw: str) -> dict[str, Any] | None:
    """Parse an LLM scoring response into a scores dict.

    Computes weighted composite_score:
        0.25 * wedge_clarity + 0.20 * technical_moat + 0.25 * market_pull
        + 0.15 * founder_fit + 0.15 * composite_synergy

    Args:
        raw: Raw JSON string from the LLM.

    Returns:
        Dict with score keys and composite_score, or None on parse failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    wedge = _int(data.get("wedge_clarity"))
    moat = _int(data.get("technical_moat"))
    pull = _int(data.get("market_pull"))
    fit = _int(data.get("founder_fit"))
    synergy = _int(data.get("composite_synergy"))
    notes = str(data.get("scoring_notes", "")).strip() or None

    composite: float | None = None
    if all(v is not None for v in [wedge, moat, pull, fit, synergy]):
        composite = round(
            0.25 * wedge + 0.20 * moat + 0.25 * pull + 0.15 * fit + 0.15 * synergy,  # type: ignore[operator]
            2,
        )

    return {
        "wedge_clarity": wedge,
        "technical_moat": moat,
        "market_pull": pull,
        "founder_fit": fit,
        "composite_synergy": synergy,
        "composite_score": composite,
        "scoring_notes": notes,
    }


def score_composite_ideas(
    ideas: list[CompositeIdea],
    papers_by_id: dict[str, "AnalyzedPaper"],
    llm_fn: Callable[[list[dict]], str],
) -> list[ScoredCompositeIdea]:
    """Score each composite idea via the provided LLM function.

    Makes up to 2 attempts per idea; falls back to a placeholder row on total failure.

    Args:
        ideas: List of composite ideas to score.
        papers_by_id: Full paper lookup dict for aggregating signals.
        llm_fn: Callable that takes a messages list and returns raw JSON string.

    Returns:
        List of ScoredCompositeIdea objects in the same order as input.
    """
    scored: list[ScoredCompositeIdea] = []

    for idea in ideas:
        signals = aggregate_paper_signals(idea.paper_ids, papers_by_id)
        messages = build_scoring_prompt(idea, signals)
        result: dict[str, Any] | None = None

        for attempt in range(2):
            try:
                raw = llm_fn(messages)
                result = parse_score_response(raw)
                if result is not None:
                    break
            except Exception as exc:
                LOGGER.warning(
                    "Scoring attempt %s failed for '%s': %s", attempt + 1, idea.title, exc
                )

        if result is not None:
            scored.append(ScoredCompositeIdea(
                id=idea.id,
                title=idea.title,
                theme_name=idea.theme_name,
                paper_ids=idea.paper_ids,
                core_capability=idea.core_capability,
                added_value=idea.added_value,
                target_user=idea.target_user,
                problem_statement=idea.problem_statement,
                wedge_description=idea.wedge_description,
                risks=idea.risks,
                wedge_clarity=result["wedge_clarity"],
                technical_moat=result["technical_moat"],
                market_pull=result["market_pull"],
                founder_fit=result["founder_fit"],
                composite_synergy=result["composite_synergy"],
                composite_score=result["composite_score"],
                scoring_notes=result["scoring_notes"],
            ))
            LOGGER.info(
                "Scored '%s': composite_score=%s",
                idea.title,
                result["composite_score"],
            )
        else:
            scored.append(ScoredCompositeIdea(
                id=idea.id,
                title=idea.title,
                theme_name=idea.theme_name,
                paper_ids=idea.paper_ids,
                core_capability=idea.core_capability,
                added_value=idea.added_value,
                target_user=idea.target_user,
                problem_statement=idea.problem_statement,
                wedge_description=idea.wedge_description,
                risks=idea.risks,
                wedge_clarity=None,
                technical_moat=None,
                market_pull=None,
                founder_fit=None,
                composite_synergy=None,
                composite_score=None,
                scoring_notes="[scoring failed]",
            ))
            LOGGER.warning("Scoring failed for '%s', using placeholder.", idea.title)

    return scored


def scored_ideas_to_rows(ideas: list[ScoredCompositeIdea]) -> list[dict]:
    """Serialize scored ideas to flat dicts, sorted by composite_score descending.

    Ideas with a score come before unscored fallback rows.
    """
    sorted_ideas = sorted(
        ideas,
        key=lambda i: (i.composite_score is not None, i.composite_score or 0.0),
        reverse=True,
    )
    rows = []
    for idea in sorted_ideas:
        rows.append({
            "id": idea.id,
            "theme_name": idea.theme_name,
            "paper_ids": json.dumps(idea.paper_ids),
            "title": idea.title,
            "core_capability": idea.core_capability,
            "added_value": idea.added_value,
            "target_user": idea.target_user,
            "problem_statement": idea.problem_statement,
            "wedge_description": idea.wedge_description,
            "risks": idea.risks,
            "wedge_clarity": idea.wedge_clarity if idea.wedge_clarity is not None else "",
            "technical_moat": idea.technical_moat if idea.technical_moat is not None else "",
            "market_pull": idea.market_pull if idea.market_pull is not None else "",
            "founder_fit": idea.founder_fit if idea.founder_fit is not None else "",
            "composite_synergy": idea.composite_synergy if idea.composite_synergy is not None else "",
            "composite_score": idea.composite_score if idea.composite_score is not None else "",
            "scoring_notes": idea.scoring_notes or "",
        })
    return rows
