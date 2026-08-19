import json

import pytest

from scripts.repair.preflight_repair_run import validate_repair_inputs


def _prepare_inputs(tmp_path, question_count=2):
    adapter = tmp_path / "adapter"
    model = tmp_path / "model"
    adapter.mkdir()
    model.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    dataset = tmp_path / "repair.json"
    dataset.write_text(
        json.dumps(
            {"examples": [{"question": f"question-{index}"} for index in range(question_count)]}
        ),
        encoding="utf-8",
    )
    return adapter, model, dataset


def test_preflight_accepts_complete_local_artifacts(tmp_path):
    adapter, model, dataset = _prepare_inputs(tmp_path)

    result = validate_repair_inputs(
        adapter_path=adapter,
        model_path=model,
        dataset_path=dataset,
        max_samples=2,
        require_cuda=False,
        check_imports=False,
    )

    assert result["status"] == "ok"
    assert result["selected_examples"] == 2
    assert result["adapter_weights"].endswith("adapter_model.safetensors")


def test_preflight_rejects_more_samples_than_dataset(tmp_path):
    adapter, model, dataset = _prepare_inputs(tmp_path, question_count=1)

    with pytest.raises(ValueError, match="contains only 1"):
        validate_repair_inputs(
            adapter_path=adapter,
            model_path=model,
            dataset_path=dataset,
            max_samples=2,
            require_cuda=False,
            check_imports=False,
        )
