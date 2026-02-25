# Simplified Environment Launcher
$repoRoot = Resolve-Path "$PSScriptRoot\.."
uv run --project "$repoRoot\projects\environment" environment $args
