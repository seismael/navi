# Simplified Dashboard Launcher
$repoRoot = Resolve-Path "$PSScriptRoot\.."
uv run --project "$repoRoot\projects\auditor" dashboard $args
