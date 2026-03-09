# Canonical Dashboard Launcher
[CmdletBinding(PositionalBinding = $false)]
param(
	[string]$PythonVersion = "3.12",
	[Parameter(ValueFromRemainingArguments = $true)]
	[string[]]$ForwardArgs
)

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$uvArgs = @(
	"run",
	"--python", $PythonVersion,
	"--project", "$repoRoot\projects\auditor",
	"navi-auditor",
	"dashboard"
)

if ($ForwardArgs) {
	$uvArgs += $ForwardArgs
}

uv @uvArgs
