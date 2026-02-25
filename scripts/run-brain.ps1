# Simplified Brain Launcher
$repoRoot = Resolve-Path "$PSScriptRoot\.."
uv run --project "$repoRoot\projects\actor" brain $args
