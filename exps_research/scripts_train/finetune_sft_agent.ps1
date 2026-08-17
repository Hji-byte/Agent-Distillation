param(
    [string]$Model = $env:AGENT_DISTILLATION_MODEL_PATH,
    [string]$DataPath = "data_processor\processed\sft\qwen35_27b_math_medium_hard_1646_v126.jsonl",
    [string]$Postfix = "qlora",
    [int]$Epochs = 2,
    [int]$DatasetSize = -1,
    [int]$DatasetStartIndex = 0,
    [int]$MaxSteps = -1,
    [int]$SaveSteps = 25,
    [string]$ResumeFromCheckpoint = ""
)

if ([string]::IsNullOrWhiteSpace($Model)) {
    throw "Model path is required. Pass -Model or set AGENT_DISTILLATION_MODEL_PATH."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainScript = Join-Path $PSScriptRoot "..\finetune_sft.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found at $python"
}
if (-not [System.IO.Path]::IsPathRooted($DataPath)) {
    $DataPath = Join-Path $projectRoot $DataPath
}
if (-not (Test-Path -LiteralPath $DataPath)) {
    throw "Training data not found: $DataPath"
}

$trainArgs = @(
    "-u",
    $trainScript,
    "--model_name", $Model,
    "--num_epochs", $Epochs,
    "--max_steps", $MaxSteps,
    "--save_steps", $SaveSteps,
    "--save_total_limit", 2,
    "--batch_size", 1,
    "--gradient_accumulation_steps", 8,
    "--lr", "2e-4",
    "--train_filepath", $DataPath,
    "--postfix", $Postfix,
    "--solution_type", "agent",
    "--use_qlora",
    "--gradient_checkpointing",
    "--lora_r", 64,
    "--lora_alpha", 128,
    "--lora_dropout", 0.05,
    "--optim", "adamw_torch_fused",
    "--max_length", 4096
)

if ($DatasetSize -gt 0) {
    $trainArgs += @(
        "--dataset_size", $DatasetSize,
        "--dataset_start_index", $DatasetStartIndex
    )
}

if ($ResumeFromCheckpoint) {
    $trainArgs += @("--resume_from_checkpoint", $ResumeFromCheckpoint)
}

Push-Location $projectRoot
try {
    & $python @trainArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
