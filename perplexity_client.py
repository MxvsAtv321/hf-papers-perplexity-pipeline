"""Perplexity API client for paper triage and startup-angle analysis."""

from __future__ import annotations

import json
import logging
import os
from json import JSONDecodeError
from typing import Any

import requests

from models import Paper

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
PERPLEXITY_TEMPERATURE = float(os.getenv("PERPLEXITY_TEMPERATURE", "0.1"))
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 2

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are my deep-tech VC partner and technical product strategist.
You will be given a ML paper (title, URL, abstract).
Analyze it and respond only with valid JSON following the provided schema, no extra text.
Do not include markdown.

Required JSON schema:
{
  "summary": {
    "problem": "",
    "core_method": "",
    "key_technical_idea": "",
    "inputs_outputs": "",
    "data_assumptions": "",
    "metrics_and_baselines": "",
    "limitations": ""
  },
  "capability": {
    "plain_language_capability": ""
  },
  "product_angles": [
    {
      "name": "",
      "target_user_persona": "",
      "problem_statement": "",
      "ten_x_improvement": "",
      "v0_demo": "",
      "technical_risks": ""
    }
  ],
  "competition": [
    {
      "name": "",
      "type": "",
      "url": "",
      "difference_vs_paper": ""
    }
  ],
  "top_bets": [
    {
      "product_angle_name": "",
      "rationale": ""
    }
  ]
}
"""


def analyze_paper(paper: Paper) -> dict[str, Any]:
    """Analyze one paper with Perplexity and return structured JSON."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY environment variable is required")

    LOGGER.info("Analyzing paper with Perplexity: %s", paper.title)
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = _call_perplexity(api_key=api_key, paper=paper)
            parsed = _parse_analysis_json(content)
            if not validate_analysis_schema(parsed):
                raise RuntimeError("Perplexity response JSON did not match required schema")
            LOGGER.info("Perplexity analysis succeeded for paper_id=%s", paper.paper_id)
            return parsed
        except Exception as exc:  # broad to preserve graceful retry path
            last_error = exc
            LOGGER.warning(
                "Perplexity analysis failed for paper_id=%s on attempt %s/%s: %s",
                paper.paper_id,
                attempt,
                MAX_ATTEMPTS,
                exc,
            )

    raise RuntimeError(f"Perplexity analysis failed for paper_id={paper.paper_id}: {last_error}")


def _call_perplexity(api_key: str, paper: Paper) -> str:
    user_prompt = (
        f"Title: {paper.title}\n"
        f"URL: {paper.url}\n"
        f"Abstract: {paper.abstract or 'Not available.'}\n"
    )

    payload = {
        "model": PERPLEXITY_MODEL,
        "temperature": PERPLEXITY_TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        PERPLEXITY_API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Perplexity response shape: {body}") from exc


def _parse_analysis_json(content: str) -> dict[str, Any]:
    """Parse possibly noisy model output into a strict JSON object."""
    try:
        parsed = json.loads(content)
    except JSONDecodeError:
        parsed = _extract_first_json_object(content)

    if not isinstance(parsed, dict):
        raise RuntimeError("Expected JSON object from Perplexity response")
    return parsed


def _extract_first_json_object(content: str) -> dict[str, Any]:
    """Extract the first decodable JSON object from an arbitrary string."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise RuntimeError("Could not extract valid JSON object from Perplexity output")


def validate_analysis_schema(data: dict[str, Any]) -> bool:
    """Basic schema validation for downstream safety."""
    required_top = {"summary", "capability", "product_angles", "competition", "top_bets"}
    if not required_top.issubset(data.keys()):
        return False

    summary = data.get("summary")
    capability = data.get("capability")
    product_angles = data.get("product_angles")
    competition = data.get("competition")
    top_bets = data.get("top_bets")

    if not isinstance(summary, dict) or not isinstance(capability, dict):
        return False
    if not isinstance(product_angles, list) or not isinstance(competition, list) or not isinstance(top_bets, list):
        return False

    required_summary = {
        "problem",
        "core_method",
        "key_technical_idea",
        "inputs_outputs",
        "data_assumptions",
        "metrics_and_baselines",
        "limitations",
    }
    if not required_summary.issubset(summary.keys()):
        return False
    if "plain_language_capability" not in capability:
        return False

    return True
