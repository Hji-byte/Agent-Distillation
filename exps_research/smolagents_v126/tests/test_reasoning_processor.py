from types import SimpleNamespace

from exps_research.unified_framework.processors.reasoning import _get_token_counts


def test_token_counts_use_current_smolagents_response_usage():
    response = SimpleNamespace(
        token_usage=SimpleNamespace(input_tokens=12, output_tokens=7)
    )

    assert _get_token_counts(SimpleNamespace(), response) == {
        "input_token_count": 12,
        "output_token_count": 7,
    }


def test_token_counts_keep_legacy_model_compatibility():
    model = SimpleNamespace(
        get_token_counts=lambda: {
            "input_token_count": 5,
            "output_token_count": 3,
        }
    )

    assert _get_token_counts(model) == {
        "input_token_count": 5,
        "output_token_count": 3,
    }
