param(
    [string]$ModelPath = $env:AGENT_DISTILLATION_MODEL_PATH,

    [string]$DataPath = "data_processor/math_dataset/test/math_500_20250414.json",

    [string]$LogFolder = "logs/qa_results/transformers/qwen3.5-0.8B_v126_baseline",

    [ValidateRange(1, 500)]
    [int]$MaxSamples = 500,

    [ValidateRange(1, 20)]
    [int]$MaxSteps = 5,

    [ValidateRange(64, 8192)]
    [int]$MaxTokens = 2048,

    [string]$Suffix = "v126_baseline",

    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python was not found at $pythonPath"
}
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    throw "Model path is required. Pass -ModelPath or set AGENT_DISTILLATION_MODEL_PATH."
}
if (-not [System.IO.Path]::IsPathRooted($DataPath)) {
    $DataPath = Join-Path $projectRoot $DataPath
}
if (-not (Test-Path -LiteralPath $DataPath)) {
    throw "Math500 dataset was not found at $DataPath"
}

$arguments = @(
    "-u", "-m", "exps_research.unified_framework.run_experiment",
    "--experiment_type", "agent",
    "--data_path", $DataPath,
    "--model_type", "transformers",
    "--model_id", $ModelPath,
    "--log_folder", $LogFolder,
    "--max_tokens", $MaxTokens,
    "--max_steps", $MaxSteps,
    "--max_samples", $MaxSamples,
    "--task_type", "math",
    "--n", "1",
    "--temperature", "0.0",
    "--seed", "42",
    "--suffix", $Suffix
)

if ($VerboseOutput) {
    $arguments += "--verbose"
}

Write-Host "Evaluating the untouched local checkpoint: $ModelPath"
Write-Host "Dataset: $DataPath ($MaxSamples samples)"
Write-Host "Generation budget: $MaxTokens tokens; hard-truncation retry disabled"
Write-Host "No SFT checkpoint or LoRA adapter will be loaded."

Push-Location $projectRoot
try {
    & $pythonPath @arguments
}
finally {
    Pop-Location
}

if ($LASTEXITCODE -ne 0) {
    throw "Local baseline evaluation failed with exit code $LASTEXITCODE."
}
