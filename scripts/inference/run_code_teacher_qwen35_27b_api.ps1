param(
    [ValidateSet("math", "math_medium", "math_retry_3", "math_failed_retry", "math_medium_failed_retry", "failed_all", "all")]
    [string]$Dataset = "math",

    [string]$ModelId = "qwen3.5-27b",

    [ValidateRange(1, 1000000)]
    [int]$MaxSamples = 1,

    [ValidateRange(1, 100)]
    [int]$ParallelWorkers = 20,

    [ValidateRange(1, 20)]
    [int]$MaxSteps = 5,

    [ValidateRange(128, 8192)]
    [int]$MaxTokens = 1280,

    [ValidateRange(128, 8192)]
    [int]$RetryMaxTokens = 2048,

    [ValidateRange(30, 3600)]
    [int]$QuestionTimeoutSeconds = 600,

    [string]$Suffix = "v126_native",

    [switch]$VerboseAgent
)

$ErrorActionPreference = "Stop"

if ($RetryMaxTokens -le $MaxTokens) {
    throw "RetryMaxTokens must be greater than MaxTokens."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python was not found at $pythonPath"
}

$datasetPaths = @{
    math = "data_processor/math_dataset/train/math_1000_20250414.json"
    math_medium = "data_processor/math_dataset/train/math_medium_1000_20250430.json"
    math_retry_3 = "data_processor/math_dataset/train/math_retry_3_20260812.json"
    math_failed_retry = "data_processor/math_dataset/train/math_hard_failed_retry_20260817.json"
    math_medium_failed_retry = "data_processor/math_dataset/train/math_medium_failed_retry_20260817.json"
}

if ($Dataset -eq "all") {
    $selectedDatasets = @("math", "math_medium")
} elseif ($Dataset -eq "failed_all") {
    $selectedDatasets = @("math_failed_retry", "math_medium_failed_retry")
} else {
    $selectedDatasets = @($Dataset)
}

foreach ($datasetName in $selectedDatasets) {
    $arguments = @(
        "-m", "exps_research.unified_framework.run_experiment",
        "--experiment_type", "agent",
        "--data_path", $datasetPaths[$datasetName],
        "--model_type", "openai",
        "--model_id", $ModelId,
        "--max_tokens", $MaxTokens,
        "--retry_max_tokens", $RetryMaxTokens,
        "--max_steps", $MaxSteps,
        "--max_samples", $MaxSamples,
        "--parallel_workers", $ParallelWorkers,
        "--multithreading",
        "--isolate_agent_processes",
        "--question_timeout_seconds", $QuestionTimeoutSeconds,
        "--n", "1",
        "--temperature", "0.0",
        "--seed", "42",
        "--do_filtering"
    )

    if ($VerboseAgent) {
        $arguments += "--verbose"
    }

    if ($Suffix) {
        $arguments += @("--suffix", $Suffix)
    }

    Write-Host "Generating $MaxSamples trajectory/trajectories from $datasetName with $ModelId (workers=$ParallelWorkers, max_tokens=$MaxTokens, retry=$RetryMaxTokens, question_timeout=${QuestionTimeoutSeconds}s)"
    Push-Location $projectRoot
    try {
        & $pythonPath @arguments
    }
    finally {
        Pop-Location
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Trajectory generation failed for $datasetName with exit code $LASTEXITCODE."
    }
}
