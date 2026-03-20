  # ── EVALUATION MODE ──
  elseif ($Evaluate) {
      if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
          Write-Host "ERROR: -Checkpoint must be specified for -Evaluate mode." -ForegroundColor Red
          exit 1
      }
      if ([string]::IsNullOrWhiteSpace($resolvedGmDagFile) -and [string]::IsNullOrWhiteSpace($Scene)) {
          Write-Host "ERROR: -Scene or -GmDagFile must be specified for -Evaluate mode." -ForegroundColor Red
          exit 1
      }
      $evalScene = if ($resolvedGmDagFile) { $resolvedGmDagFile } else { $Scene }

      $evalArgs = @(
          "run",
          "--python", $PythonVersion,
          "--project", (Join-Path $repoRoot "projects\actor"),
          "python", "-m", "navi_actor.cli", "evaluate",
          "--checkpoint", $Checkpoint,
          "--scene", $evalScene,
          "--actors", "$NumActors",
          "--total-steps", $TotalSteps,
          "--actor-pub", "tcp://*:$ActorTelemetryPort",
          "--azimuth-bins", "$AzimuthBins",
          "--elevation-bins", "$ElevationBins"
      )

      $evalLogOut = Join-Path $logsDir "evaluate.out.log"
      $evalLogErr = Join-Path $logsDir "evaluate.err.log"
      $evalUnifiedLog = Join-Path $repoRoot "logs\navi_actor_evaluate.log"

      $evalProc = $null
      try {
          Write-Host "========================================================"
          Write-Host "  Navi Ghost-Matrix Evaluation"
          Write-Host "  Scene      : $evalScene"
          Write-Host "  Checkpoint : $Checkpoint"
          Write-Host "  Actors     : $NumActors"
          Write-Host "  Telemetry  : tcp://localhost:$ActorTelemetryPort"
          Write-Host "========================================================"

          Write-Host "`nStarting canonical evaluate (background)..."
          $evalProc = Start-BackgroundUv -RepoRoot $repoRoot -UvArgs $evalArgs -StdOutFile $evalLogOut -StdErrFile $evalLogErr
          Write-Host "  PID: $($evalProc.Id)"
          Write-Host "  Logs: $evalLogOut"
          Write-Host "        $evalLogErr"
          
          if ($dashboardEnabled) {
              Write-Host "`nStarting Auditor Dashboard..."
              $dashArgs = @(
                  "run",
                  "--python", $PythonVersion,
                  "--project", (Join-Path $repoRoot "projects\auditor"),
                  "python", "-m", "navi_auditor.dashboard.app",
                  "--actor-control", "tcp://localhost:$ActorControlPort",
                  "--actor-pub", "tcp://localhost:$ActorTelemetryPort",
                  "--passive"
              )
              Start-Process -FilePath "uv" -ArgumentList $dashArgs -WorkingDirectory $repoRoot
          }

          Write-Host "  Tail logs: Get-Content '$evalLogOut' -Wait"
          Write-Host "  Stop:      Stop-Process -Id $($evalProc.Id) -Force"
          Wait-Process -Id $evalProc.Id
      }
      finally {
          if ($null -ne $evalProc -and -not $evalProc.HasExited) {
              Write-Host "`nStopping canonical evaluate (PID $($evalProc.Id))..."
              Stop-ProcessTreeById -ProcessId $evalProc.Id
          }
      }
  }