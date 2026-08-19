import json

from scripts.repair.generate_local_repairs import (
    write_run_manifest,
    load_completed,
    truncate_incomplete_jsonl_tail,
)


def test_truncate_incomplete_jsonl_tail_preserves_complete_records(tmp_path):
    path = tmp_path / "attempts.jsonl"
    complete = json.dumps({"repair_id": "done"}).encode("utf-8") + b"\n"
    path.write_bytes(complete + b'{"repair_id":"partial"')

    assert truncate_incomplete_jsonl_tail(path)
    assert path.read_bytes() == complete
    assert load_completed(path) == {"done"}
    assert not truncate_incomplete_jsonl_tail(path)


def test_truncate_only_incomplete_record_empties_file(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b'{"repair_id":"partial"')
    assert truncate_incomplete_jsonl_tail(path)
    assert path.read_bytes() == b""


def test_retryable_attempt_is_not_treated_as_completed(tmp_path):
    path = tmp_path / "attempts.jsonl"
    rows = [
        {"repair_id": "retry-me", "retryable_error": True, "accepted": False},
        {"repair_id": "done", "accepted": False},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert load_completed(path) == {"done"}


def test_run_manifest_refuses_configuration_drift(tmp_path):
    path = tmp_path / "run_manifest.json"
    write_run_manifest(path, {"run_tag": "v2", "seed": 42})
    write_run_manifest(path, {"run_tag": "v2", "seed": 42})

    try:
        write_run_manifest(path, {"run_tag": "v2", "seed": 7})
    except ValueError as error:
        assert "different settings" in str(error)
    else:
        raise AssertionError("configuration drift should be rejected")
