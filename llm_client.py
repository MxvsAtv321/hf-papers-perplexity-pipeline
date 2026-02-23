"""OpenAI GPT-based LLM client for paper analysis."""

from __future__ import annotations

import json
import logging
import os
from json import JSONDecodeError
from typing import Any

from openai import OpenAI

from models import Paper

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
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
    """Analyze one paper with OpenAI and return structured JSON."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    LOGGER.info("Analyzing paper with OpenAI: %s", paper.title)
    last_error: Exception | None = None

    client = OpenAI(api_key=api_key)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            parsed = _call_openai(client=client, paper=paper)
            if not validate_analysis_schema(parsed):
                raise RuntimeError("OpenAI response JSON did not match required schema")
            LOGGER.info("OpenAI analysis succeeded for paper_id=%s", paper.paper_id)
            return parsed
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "OpenAI analysis failed for paper_id=%s on attempt %s/%s: %s",
                paper.paper_id,
                attempt,
                MAX_ATTEMPTS,
                exc,
            )

    raise RuntimeError(f"OpenAI analysis failed for paper_id={paper.paper_id}: {last_error}")


def _call_openai(client: OpenAI, paper: Paper) -> dict[str, Any]:
    user_prompt = (
        f"Title: {paper.title}\n"
        f"URL: {paper.url}\n"
        f"Abstract: {paper.abstract or 'Not available.'}\n"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response")

    return _parse_analysis_json(content)


def _parse_analysis_json(content: str) -> dict[str, Any]:
    """Parse possibly noisy model output into a strict JSON object."""
    try:
        parsed = json.loads(content)
    except JSONDecodeError:
        parsed = _extract_first_json_object(content)

    if not isinstance(parsed, dict):
        raise RuntimeError("Expected JSON object from OpenAI response")
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
    raise RuntimeError("Could not extract valid JSON object from OpenAI output")


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
