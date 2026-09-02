param(
    [ValidateSet("build", "setup", "teacher-one", "teacher-five", "teacher-50", "teacher-500", "teacher-smoke", "teacher-full", "check")]
    [string]$Action = "check"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$ComposeFile = Join-Path $ProjectRoot "docker/alfworld/compose.yaml"
$EnvFile = Join-Path $ProjectRoot ".env"

switch ($Action) {
    "build" {
        docker compose -f $ComposeFile build
    }
    "setup" {
        docker compose -f $ComposeFile run --rm alfworld bash scripts/alfworld/setup_eto_alfworld.sh eval
    }
    "teacher-smoke" {
        if (-not (Test-Path $EnvFile)) {
            throw "Missing project .env file with ALFWORLD_API_KEY."
        }
        docker compose --env-file $EnvFile -f $ComposeFile run --rm alfworld bash scripts/alfworld/run_eto_alfworld_teacher.sh train smoke
    }
    "teacher-one" {
        if (-not (Test-Path $EnvFile)) {
            throw "Missing project .env file with ALFWORLD_API_KEY."
        }
        docker compose --env-file $EnvFile -f $ComposeFile run --rm alfworld bash scripts/alfworld/run_eto_alfworld_teacher.sh train one
    }
    "teacher-five" {
        if (-not (Test-Path $EnvFile)) {
            throw "Missing project .env file with ALFWORLD_API_KEY."
        }
        docker compose --env-file $EnvFile -f $ComposeFile run --rm alfworld bash scripts/alfworld/run_eto_alfworld_teacher.sh train five
    }
    "teacher-50" {
        if (-not (Test-Path $EnvFile)) {
            throw "Missing project .env file with ALFWORLD_API_KEY."
        }
        docker compose --env-file $EnvFile -f $ComposeFile run --rm alfworld bash scripts/alfworld/run_eto_alfworld_teacher.sh train fifty
    }
    "teacher-500" {
        if (-not (Test-Path $EnvFile)) {
            throw "Missing project .env file with ALFWORLD_API_KEY."
        }
        docker compose --env-file $EnvFile -f $ComposeFile run --rm alfworld bash scripts/alfworld/run_eto_alfworld_teacher.sh train five-hundred
    }
    "teacher-full" {
        if (-not (Test-Path $EnvFile)) {
            throw "Missing project .env file with ALFWORLD_API_KEY."
        }
        docker compose --env-file $EnvFile -f $ComposeFile run --rm alfworld bash scripts/alfworld/run_eto_alfworld_teacher.sh train full
    }
    "check" {
        docker compose -f $ComposeFile run --rm alfworld python -m exps_research.alfworld_eto.integrity /workspace/_local/upstream/ETO
    }
}
