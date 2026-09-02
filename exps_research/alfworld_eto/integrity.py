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
    "eval_agent/main.py": "04fd0ef3097e1c29c2e3065d589226594b6ead0984de9bdb39b891cb23005bbe",
    "eval_agent/agents/base.py": "af3ea2732a81a2b6dde709a39f5e871789c0201ab5ff8803d51f54f117f898b2",
    "eval_agent/agents/openai_lm_agent.py": "fda87bf2ae6371e5f9947e5c6ee90c0a3bdd78b0c427a74ea35fe63871e28d1e",
    "eval_agent/envs/alfworld_env.py": "84b5c69bf44a91107e2a99d2b4d6119f4441474ba8317d7a79b9bdb5691ad026",
    "eval_agent/tasks/alfworld.py": "5db72e36d0c37a3ea246397a8c2f581b30b701ee5b12ef0f025f213e7b3a5125",
    "eval_agent/prompt/templates.py": "6ef8c35ee92daff6f1f8e0e12a3e581384de05ab5630c59d75d774013022f702",
    "eval_agent/prompt/instructions/alfworld_inst.txt": "ed99085b2fcef955363a031d274e9ffcc7a312c8350776f7464b4a8fb9ae85d7",
    "eval_agent/prompt/icl_examples/alfworld_icl.json": "06b00f15c320c484d86fffcfaedcd057eb7cca544b452683cac9f8b9a5d2ff55",
    "eval_agent/configs/task/alfworld.json": "9f9bcdafdf84a550be30b926b597ebe196b5a675804f4f50b6d4eef37ec52190",
    "eval_agent/data/alfworld/base_config.yaml": "d95e445ae3b3fc1f277a6ebd613678bf2c46a39a4975d8a12a85d642e8d40016",
}

DATA_DIRECTORIES = (
    "eval_agent/data/alfworld/json_2.1.1/train",
    "eval_agent/data/alfworld/json_2.1.1/valid_seen",
    "eval_agent/data/alfworld/json_2.1.1/valid_unseen",
)


def sha256_file(path: Path) -> str:
    # Git may check text files out with CRLF on Windows and LF on Linux.
    # Normalize line endings so integrity protects content across platforms.
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


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
