<#
.SYNOPSIS
    Behavioral Cloning pre-training from recorded human demonstrations.
.DESCRIPTION
    Trains the CognitiveMambaPolicy via supervised learning on demonstration
    recordings created with ``uv run explore --record``.  Produces a v2
    checkpoint compatible with ``uv run brain train --checkpoint <path>``.

    Workflow:
      1. Record: uv run explore --record
      2. Pre-train: .\scripts\run-bc-pretrain.ps1
      3. Fine-tune: .\scripts\train.ps1 -ResumeCheckpoint artifacts\checkpoints\bc_base_model.pt
#>
param(
    [string]$Demonstrations = "artifacts/demonstrations",
    [string]$Output = "artifacts/checkpoints/bc_base_model.pt",
    [string]$Checkpoint = "",
    [int]$Epochs = 50,
    [float]$LearningRate = 1e-3,
    [int]$BpttLen = 8,
    [int]$MinibatchSize = 32,
    [string]$TemporalCore = ""
)

$ErrorActionPreference = "Continue"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) { $PSNativeCommandUseErrorActionPreference = $false }

Set-Location $PSScriptRoot\..

Write-Host "=== Ghost-Matrix Behavioral Cloning Pre-Training ===" -ForegroundColor Cyan
Write-Host "  Demonstrations: $Demonstrations"
Write-Host "  Output:         $Output"
Write-Host "  Checkpoint:     $(if ($Checkpoint) { $Checkpoint } else { '(fresh start)' })"
Write-Host "  Epochs:         $Epochs"
Write-Host ""

$uvArgs = @(
    "run", "--project", "projects/actor",
    "brain", "bc-pretrain",
    "--demonstrations", $Demonstrations,
    "--output", $Output,
    "--epochs", $Epochs,
    "--learning-rate", $LearningRate,
    "--bptt-len", $BpttLen,
    "--minibatch-size", $MinibatchSize
)

if ($TemporalCore) {
    $uvArgs += @("--temporal-core", $TemporalCore)
}

if ($Checkpoint) {
    $uvArgs += @("--checkpoint", $Checkpoint)
}

& uv @uvArgs 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Host $_.Exception.Message } else { $_ } }

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== BC Pre-Training Complete ===" -ForegroundColor Green
    Write-Host "  Checkpoint: $Output"
    Write-Host ""
    Write-Host "  Next step - fine-tune with RL:"
    Write-Host "    .\scripts\train.ps1 -ResumeCheckpoint $Output"
} else {
    Write-Host "BC pre-training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
