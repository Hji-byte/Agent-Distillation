#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
cd "$project_root"

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    python_bin="${PYTHON:-python}"
fi

eto_root="${ETO_ROOT:-$project_root/_local/upstream/ETO}"
asset_mode="${1:-eval}"
if [[ "$asset_mode" != "eval" && "$asset_mode" != "training" ]]; then
    echo "Asset mode must be 'eval' or 'training'." >&2
    exit 2
fi
mkdir -p "$(dirname "$eto_root")"

if [[ ! -d "$eto_root/.git" ]]; then
    git clone https://github.com/Yifan-Song793/ETO.git "$eto_root"
    git -C "$eto_root" checkout --detach a2fc5da38f8d00cfaf3f9b6370d586eebaf72904
fi

"$python_bin" -m exps_research.alfworld_eto.integrity "$eto_root"

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python_bin" -r requirements/alfworld-eto.txt
else
    "$python_bin" -m ensurepip --upgrade
    "$python_bin" -m pip install -r requirements/alfworld-eto.txt
fi

data_dir="$eto_root/eval_agent/data/alfworld"
if [[ ! -d "$data_dir/json_2.1.1/train" ]]; then
    download_dir="$project_root/_local/downloads"
    mkdir -p "$download_dir"
    archive="$download_dir/alfworld_data.zip"
    "$python_bin" -m gdown "https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI" -O "$archive"
    "$python_bin" -m zipfile -e "$archive" "$data_dir"
fi

if [[ "$asset_mode" == "training" && ! -f "$eto_root/data/alfworld_sft.json" ]]; then
    download_dir="$project_root/_local/downloads"
    mkdir -p "$download_dir"
    training_archive="$download_dir/ETO_data.zip"
    "$python_bin" -m gdown "https://drive.google.com/uc?id=1YbhbL8RhQGDWFv5y6k1qgwRqSyFFsao8" -O "$training_archive"
    "$python_bin" -m zipfile -e "$training_archive" "$eto_root"
fi

"$python_bin" -m exps_research.alfworld_eto.integrity "$eto_root" --require-data

echo "ETO ALFWorld runtime is ready at $eto_root"
