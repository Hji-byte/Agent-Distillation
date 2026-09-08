import importlib.util
import json
import subprocess
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "inference"
    / "batched_transformers_reasoning.py"
)
SPEC = importlib.util.spec_from_file_location("batched_transformers_reasoning", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_collect_resume_records_prefers_output_and_uses_dataset_order(tmp_path):
    examples = [{"question": "q1"}, {"question": "q2"}, {"question": "q3"}]
    output = tmp_path / "output.jsonl"
    resume = tmp_path / "resume.jsonl"
    _write_jsonl(output, [{"question": "q2", "generated_answer": "new"}])
    _write_jsonl(
        resume,
        [
            {"question": "q1", "generated_answer": "old-1"},
            {"question": "q2", "generated_answer": "old-2"},
            {"question": "q3", "generated_answer": None, "error": "failed"},
            {"question": "unrelated", "generated_answer": "ignore"},
        ],
    )

    records = MODULE.collect_resume_records(examples, output, [resume])

    assert [record["question"] for record in records] == ["q1", "q2"]
    assert records[1]["generated_answer"] == "new"


def test_count_generated_tokens_includes_first_eos():
    assert MODULE.count_generated_tokens([10, 11, 2, 2], {2}) == 3
    assert MODULE.count_generated_tokens([10, 11], {2}) == 2
    assert MODULE.count_generated_tokens([], {2}) == 0


def test_adapter_argument_is_optional():
    arguments = ["--dataset", "data.json", "--model", "base", "--output", "out.jsonl"]
    assert MODULE.build_parser().parse_args(arguments).adapter is None
    parsed = MODULE.build_parser().parse_args(arguments + ["--adapter", "sft/final"])
    assert parsed.adapter == Path("sft/final")


def test_direct_script_execution_adds_project_root_to_import_path(tmp_path):
    code = (
        "import runpy,sys; "
        f"ns=runpy.run_path({str(MODULE_PATH)!r}); "
        "print(str(ns['PROJECT_ROOT']) in sys.path)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "True"
