from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "a2fc5da38f8d00cfaf3f9b6370d586eebaf72904"

# These are the upstream files that define the ALFWorld prompt, agent loop,
# environment behavior, and task split. The adapter refuses to run if any of
# them changes, so an experiment cannot silently drift from ETO.
PROTECTED_FILES = {
    "eval_agent/main.py": "dd1e3d698c9331e13dfed47c1846accf341abad0505c5807841eead62005d396",
    "eval_agent/agents/base.py": "69724877b39b45f05b95e182303cc30168badf020a731457178b29e2bcc466dc",
    "eval_agent/agents/openai_lm_agent.py": "659b88c205628f95e15264ab4a573f2c3870beaca14bd68b7f3563afb1ec053f",
    "eval_agent/envs/alfworld_env.py": "f7a2557cde8bab40c84180fc3940d2bb210d17a326a237c2c0d358f3bff95ff2",
    "eval_agent/tasks/alfworld.py": "f4f5c0da5f182e1889303f03093cf850fbc3120d2994bb0f8cb1502cc67c1d20",
    "eval_agent/prompt/templates.py": "f2204ef5d7556ebabd8e3b5fc9b73b13a3659e7bb1c2ffb3d61a6c39d4f66f5c",
    "eval_agent/prompt/instructions/alfworld_inst.txt": "91cc0e094d79f493790feb3412cc7e742cbe9ff407c0d58004fc72178f076f1a",
    "eval_agent/prompt/icl_examples/alfworld_icl.json": "e7280370ceec44a5d929ef26ccf33136d61f3d26d575c1a0edd091efb3d65cb0",
    "eval_agent/configs/task/alfworld.json": "27a4faa6cc773f14d40948a5942159a27ee106b360c29174a3c38697e1f59e3b",
    "eval_agent/data/alfworld/base_config.yaml": "ca587cad8c2683b4fb48ed6704ae0daf6037b8f323565d71d8cd12180ff0f700",
}

DATA_DIRECTORIES = (
    "eval_agent/data/alfworld/json_2.1.1/train",
    "eval_agent/data/alfworld/json_2.1.1/valid_seen",
    "eval_agent/data/alfworld/json_2.1.1/valid_unseen",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream(eto_root: Path, *, require_data: bool = False) -> dict:
    eto_root = eto_root.expanduser().resolve()
    errors: list[str] = []

    if not eto_root.is_dir():
        return {"status": "error", "eto_root": str(eto_root), "errors": ["ETO root does not exist"]}

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(eto_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        commit = None
        errors.append(f"Unable to read ETO git commit: {exc}")

    if commit != EXPECTED_COMMIT:
        errors.append(f"ETO commit mismatch: expected {EXPECTED_COMMIT}, found {commit}")

    checked_files: dict[str, str] = {}
    for relative_path, expected_hash in PROTECTED_FILES.items():
        path = eto_root / relative_path
        if not path.is_file():
            errors.append(f"Missing protected upstream file: {relative_path}")
            continue
        actual_hash = sha256_file(path)
        checked_files[relative_path] = actual_hash
        if actual_hash != expected_hash:
            errors.append(
                f"Protected upstream file changed: {relative_path} "
                f"(expected {expected_hash}, found {actual_hash})"
            )

    missing_data = [relative for relative in DATA_DIRECTORIES if not (eto_root / relative).is_dir()]
    if require_data and missing_data:
        errors.append("ALFWorld runtime data is missing; run scripts/alfworld/setup_eto_alfworld.sh")

    return {
        "status": "ok" if not errors else "error",
        "eto_root": str(eto_root),
        "commit": commit,
        "expected_commit": EXPECTED_COMMIT,
        "protected_files_checked": len(checked_files),
        "missing_data_directories": missing_data,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the untouched ETO ALFWorld source and assets")
    parser.add_argument("eto_root", type=Path)
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()

    report = verify_upstream(args.eto_root, require_data=args.require_data)
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
