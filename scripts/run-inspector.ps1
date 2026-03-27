<#
.SYNOPSIS
    Launch the GMDAG 3D Inspector viewer.

.DESCRIPTION
    Opens the interactive PyVista-based 3D inspector for .gmdag files.
    Extracts SDF isosurface via marching cubes and presents it for
    interactive orbit/pan/zoom navigation with keyboard shortcuts.

.PARAMETER GmdagFile
    Path to a .gmdag file to inspect.

.PARAMETER Resolution
    Initial extraction resolution (default: 128 for instant preview).

.PARAMETER Action
    CLI action: view (default), info, export, corpus.

.EXAMPLE
    .\scripts\run-inspector.ps1 -GmdagFile artifacts\gmdag\corpus\ai-habitat_habitat_test_scenes\apartment_1.gmdag
    .\scripts\run-inspector.ps1 -GmdagFile scene.gmdag -Resolution 256
    .\scripts\run-inspector.ps1 -Action info -GmdagFile scene.gmdag
    .\scripts\run-inspector.ps1 -Action corpus
#>

param(
    [string]$GmdagFile = "",
    [int]$Resolution = 128,
    [ValidateSet("view", "info", "export", "corpus")]
    [string]$Action = "view"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Join-Path $PSScriptRoot ".." "projects" "inspector"

Push-Location $ProjectDir
try {
    switch ($Action) {
        "view" {
            if (-not $GmdagFile) {
                Write-Error "GmdagFile is required for 'view' action."
                exit 1
            }
            $resolved = (Resolve-Path $GmdagFile -ErrorAction Stop).Path
            Write-Host "[Inspector] Launching viewer: $resolved (${Resolution}^3)" -ForegroundColor Cyan
            uv run navi-inspector view $resolved --resolution $Resolution
        }
        "info" {
            if (-not $GmdagFile) {
                Write-Error "GmdagFile is required for 'info' action."
                exit 1
            }
            $resolved = (Resolve-Path $GmdagFile -ErrorAction Stop).Path
            uv run navi-inspector info $resolved
        }
        "export" {
            if (-not $GmdagFile) {
                Write-Error "GmdagFile is required for 'export' action."
                exit 1
            }
            $resolved = (Resolve-Path $GmdagFile -ErrorAction Stop).Path
            $outName = [System.IO.Path]::GetFileNameWithoutExtension($resolved) + "_export.ply"
            $outPath = Join-Path $PSScriptRoot ".." "artifacts" "inspector" "exports" $outName
            Write-Host "[Inspector] Exporting: $resolved -> $outPath" -ForegroundColor Cyan
            uv run navi-inspector export $resolved $outPath --resolution $Resolution
        }
        "corpus" {
            uv run navi-inspector corpus
        }
    }
} finally {
    Pop-Location
}
