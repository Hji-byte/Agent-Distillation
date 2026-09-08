# Independent AutoDL GRPO environment

CURRENT PLAN: reuse the AutoDL image, not the full-isolation alternative below.
From repository root using the image's Python 3.12:

```bash
python -m venv --system-site-packages /root/autodl-tmp/envs/math-grpo-overlay
source /root/autodl-tmp/envs/math-grpo-overlay/bin/activate
python environments/math-grpo/setup_overlay.py
python environments/math-grpo/setup_overlay.py --install
```

First preview, then install. The script uses pip's JSON report to reject changes
to Torch/CUDA packages and installs only the vetted dependency list with no-deps.
It records the image inventory and final environment under the overlay's
`grpo_environment_audit/`. No image package is removed. Do NOT use uv sync with
this plan; reproducibility depends on the image as well as the overlay inventory.
Actual AutoDL install/GPU training is still untested. Files must be transferred first.

The remainder documents the UNUSED full-isolation alternative.

STATUS: configuration validated, but `uv.lock` has NOT yet been generated.
Local online resolution timed out against PyPI, Tsinghua and Aliyun on 2026-09-08.
Do not treat this directory as a locked/reproducible environment until a successful
`uv lock --project environments/math-grpo --python 3.12` and lock review. The
`--locked` installation commands below are for AFTER that step succeeds.

This is a separate uv project, not a workspace member of the legacy SFT project.
Target: Linux x86_64, Python 3.12, Torch 2.8.0+cu128. TRL and model dependencies
are pinned here; transitive versions and hashes are recorded in `uv.lock`.
No vLLM, bitsandbytes, torchvision, or old agent framework is required.

From the repository root on AutoDL:

```bash
uv sync --project environments/math-grpo --python 3.12 --locked
```

uv environments do not necessarily contain pip; use `uv pip` for checks.

```bash
uv pip check --python environments/math-grpo/.venv/bin/python
environments/math-grpo/.venv/bin/python -c "import torch, trl, transformers, peft; from trl import GRPOConfig, GRPOTrainer; from exps_research.unified_framework.math_utils.qwen_math_grader import math_equal; print(torch.__version__, trl.__version__, transformers.__version__, peft.__version__); print(math_equal('2','2')); print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16,requires_grad=True); (x@x).float().square().mean().backward(); torch.cuda.synchronize(); print('BF16 GPU check OK')"
```

Always invoke the interpreter by the explicit path above, from repository root.
Do not use root `uv sync`, `uv sync --active`, or root `uv run`: those may resolve
the old SFT project instead. No root editable install is needed for module entrypoints.

The isolated environment does NOT inherit the image's site-packages. Torch may
need downloading again; the image installation and root SFT `.venv` are untouched.
PyTorch wheels use its official cu128 index, other wheels use PyPI. A PyPI mirror
cannot automatically accelerate PyTorch wheel downloads. Do not alter the index
configuration or regenerate locks on the server without reviewing the resulting diff.

`--locked` refuses to silently change the resolution. Commit/copy both TOML and
lock file together. Creating the lock and a Linux-target dry run on Windows is
not proof of a successful GPU install: execute the checks and then a 4096-token
smoke on the actual server before a pilot. Driver support is verified on the server.

The training/scoring adapter scripts and original train data must also be copied
to the server; creating this environment does not publish local uncommitted files.
