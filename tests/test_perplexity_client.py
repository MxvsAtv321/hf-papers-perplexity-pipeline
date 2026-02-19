from perplexity_client import _parse_analysis_json, validate_analysis_schema


def test_parse_analysis_json_with_wrapping_text() -> None:
    wrapped = (
        "Here is your result:\n"
        '{"summary":{"problem":"","core_method":"","key_technical_idea":"","inputs_outputs":"",'
        '"data_assumptions":"","metrics_and_baselines":"","limitations":""},'
        '"capability":{"plain_language_capability":""},"product_angles":[],'
        '"competition":[],"top_bets":[]}\n'
        "Thanks!"
    )

    parsed = _parse_analysis_json(wrapped)
    assert isinstance(parsed, dict)
    assert validate_analysis_schema(parsed) is True
