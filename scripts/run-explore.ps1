# Manual Explorer — launches environment + dashboard with keyboard navigation
[CmdletBinding(PositionalBinding = $false)]
param(
	[string]$PythonVersion = "3.12",
	[string]$GmdagFile = "",
	[float]$LinearSpeed = 1.5,
	[float]$YawRate = 1.5,
	[Parameter(ValueFromRemainingArguments = $true)]
	[string[]]$ForwardArgs
)

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$uvArgs = @(
	"run",
	"--python", $PythonVersion,
	"--project", "$repoRoot\projects\auditor",
	"navi-auditor",
	"explore",
	"--linear-speed", $LinearSpeed,
	"--yaw-rate", $YawRate
)

if ($GmdagFile) {
	$uvArgs += @("--gmdag-file", $GmdagFile)
}

if ($ForwardArgs) {
	$uvArgs += $ForwardArgs
}

uv @uvArgs
