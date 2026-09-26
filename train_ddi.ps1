<#
.SYNOPSIS
    DDI Training Workflow Launcher
    Trains both pipelines and prints a unified performance report.

.DESCRIPTION
    Pipeline 1 (Static):   Morgan fingerprint multi-label LR on multilabel_frequent363_dataset.csv
    Pipeline 2a (Temporal): Temporal admission feature multi-label LR on temporal_multilabel_frequent363_dataset.csv
    Pipeline 2b (Ablation): Temporal feature ablation study (4 configurations)

    Usage:
        # Run all pipelines (default)
        .\train_ddi.ps1

        # Run a specific pipeline only
        .\train_ddi.ps1 -Pipeline static
        .\train_ddi.ps1 -Pipeline temporal
        .\train_ddi.ps1 -Pipeline ablation

        # Custom hyperparameters
        .\train_ddi.ps1 -Epochs 300 -LR 0.01

        # Dry run: show commands without executing
        .\train_ddi.ps1 -DryRun

.PARAMETER Pipeline
    Which pipeline(s) to run: 'all' (default), 'static', 'temporal', or 'ablation'.

.PARAMETER Epochs
    Number of training epochs (default: 200).

.PARAMETER LR
    Learning rate (default: 0.05).

.PARAMETER Seed
    Random seed (default: 42).

.PARAMETER DryRun
    Print commands without executing them.
#>

param(
    [ValidateSet("all", "static", "temporal", "ablation", "static_baseline", "static_gat", "molecular_gnn", "temporal_baseline", "temporal_gnn")]
    [string]$Pipeline = "all",

    [int]$Epochs = 200,
    [double]$LR = 0.05,
    [double]$WeightDecay = 0.001,
    [int]$Seed = 42,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# Utility functions
# ============================================================

function Write-Header {
    param([string]$Title)
    $line = "=" * 60
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host ("  " + $Title) -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("--- " + $Title + " ---") -ForegroundColor Yellow
}

function Run-Step {
    param(
        [string]$Label,
        [string[]]$Cmd
    )
    Write-Section "Running: $Label"
    $cmdStr = ($Cmd -join " ")
    Write-Host "Command: $cmdStr" -ForegroundColor DarkGray

    if ($DryRun) {
        Write-Host "[DRY RUN] Skipping execution." -ForegroundColor Magenta
        return $true
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)]
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)

    if ($exitCode -ne 0) {
        Write-Host ""
        Write-Host "[FAILED] $Label (exit code $exitCode, ${elapsed}s)" -ForegroundColor Red
        return $false
    }

    Write-Host ""
    Write-Host "[OK] $Label completed in ${elapsed}s" -ForegroundColor Green
    return $true
}

function Read-JsonSafe {
    param([string]$Path)
    if (Test-Path $Path) {
        try {
            return Get-Content $Path -Raw | ConvertFrom-Json
        } catch {
            return $null
        }
    }
    return $null
}

function Format-Metric {
    param($Value, [int]$Decimals = 4)
    if ($null -eq $Value) { return "N/A" }
    return ([math]::Round([double]$Value, $Decimals)).ToString("F$Decimals")
}

# ============================================================
# Configuration
# ============================================================

$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }

$StaticResultsJson   = Join-Path $ProjectRoot "results\metrics\static_multilabel_results.json"
$TemporalResultsJson = Join-Path $ProjectRoot "results\metrics\temporal_baseline_results.json"
$AblationResultsJson = Join-Path $ProjectRoot "results\metrics\temporal_feature_ablation_results.json"

$StaticReportMd   = Join-Path $ProjectRoot "reports\static_multilabel_evaluation.md"
$TemporalReportMd = Join-Path $ProjectRoot "reports\temporal_baseline_evaluation.md"
$AblationReportMd = Join-Path $ProjectRoot "reports\temporal_feature_ablation_report.md"

# ============================================================
# Banner
# ============================================================

Write-Header "DDI TRAINING WORKFLOW LAUNCHER"
Write-Host "  Project Root:  $ProjectRoot"
Write-Host "  Pipeline:      $Pipeline"
Write-Host "  Epochs:        $Epochs"
Write-Host "  Learning Rate: $LR"
Write-Host "  Weight Decay:  $WeightDecay"
Write-Host "  Seed:          $Seed"
if ($DryRun) {
    Write-Host "  Mode:          DRY RUN (no training will occur)" -ForegroundColor Magenta
}
Write-Host ""

$results = @{}
$failures = @()

# ============================================================
# Pipeline 1 — Static Multi-Label Baseline
# ============================================================

if ($Pipeline -eq "all" -or $Pipeline -eq "static") {
    $ok = Run-Step "Pipeline 1: Static Fingerprint Multi-Label LR" @(
        "python", "-m", "src.models.run_multilabel_pipeline",
        "--epochs", "$Epochs",
        "--lr", "$LR",
        "--weight-decay", "$WeightDecay",
        "--seed", "$Seed"
    )
    if (-not $ok) { $failures += "Pipeline 1 (Static)" }
    $results["static"] = Read-JsonSafe $StaticResultsJson
}

# ============================================================
# Pipeline 2a — Temporal Multi-Label Baseline
# ============================================================

if ($Pipeline -eq "all" -or $Pipeline -eq "temporal") {
    $ok = Run-Step "Pipeline 2a: Temporal Admission Feature Multi-Label LR" @(
        "python", "-m", "src.models.train_temporal_baseline",
        "--epochs", "$Epochs",
        "--lr", "$LR",
        "--weight-decay", "$WeightDecay",
        "--seed", "$Seed"
    )
    if (-not $ok) { $failures += "Pipeline 2a (Temporal Baseline)" }
    $results["temporal"] = Read-JsonSafe $TemporalResultsJson
}

# ============================================================
# Pipeline 2b — Temporal Feature Ablation Study
# ============================================================

if ($Pipeline -eq "all" -or $Pipeline -eq "ablation") {
    $ok = Run-Step "Pipeline 2b: Temporal Feature Ablation Study (4 configs)" @(
        "python", "-m", "src.models.run_temporal_ablation",
        "--epochs", "$Epochs",
        "--lr", "$LR",
        "--weight-decay", "$WeightDecay",
        "--seed", "$Seed"
    )
    if (-not $ok) { $failures += "Pipeline 2b (Ablation)" }
    $results["ablation"] = Read-JsonSafe $AblationResultsJson
}

# ============================================================
# UNIFIED PERFORMANCE REPORT
# ============================================================

Write-Header "DDI TRAINING PERFORMANCE REPORT"

# ------ Pipeline 1 Summary ------
if ($results.ContainsKey("static") -and $null -ne $results["static"]) {
    $s = $results["static"]
    $t = $s.model_evaluation.test_metrics_optimized_threshold
    $p = $s.prior_baseline_evaluation.test_metrics_optimized_threshold
    $fc = $s.feature_config
    $thr = $s.model_evaluation.val_threshold_optimization.best_threshold
    $obs = $s.dataset_observations
    $ts  = $s.training_summary

    Write-Section "PIPELINE 1 — Static Multi-Label Baseline (363 classes)"
    Write-Host "  Dataset:       multilabel_frequent363_dataset.csv" -ForegroundColor Gray
    Write-Host "  Split:         Train=$($obs.train) / Val=$($obs.val) / Test=$($obs.test)"
    Write-Host "  Features:      Morgan FP ($($fc.fingerprint_bits) bits, r=$($fc.fingerprint_radius)) + Age = dim $($fc.input_dim)"
    Write-Host "  Best Epoch:    $($ts.best_epoch) / $($ts.total_epochs)  |  Val Loss: $(Format-Metric $ts.best_val_loss 6)"
    Write-Host "  Val Threshold: $thr (optimized on validation set)"
    Write-Host ""
    Write-Host "  Test Results (Static LR | val-optimized threshold $thr):" -ForegroundColor White
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f "Model","Micro-F1","Macro-F1","Hamm.Loss","mAP","P@5","R@5")
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f ("="*18),("="*10),("="*10),("="*12),("="*8),("="*8),("="*8))
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f "Static FP LR",(Format-Metric $t.micro_f1),(Format-Metric $t.macro_f1),(Format-Metric $t.hamming_loss),(Format-Metric $t.mAP),(Format-Metric $t.precision_at_5),(Format-Metric $t.recall_at_5)) -ForegroundColor Green
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f "Training Prior",(Format-Metric $p.micro_f1),(Format-Metric $p.macro_f1),(Format-Metric $p.hamming_loss),(Format-Metric $p.mAP),(Format-Metric $p.precision_at_5),(Format-Metric $p.recall_at_5))
    Write-Host ""
    Write-Host "  Outputs:"
    Write-Host "    JSON:   $StaticResultsJson"
    Write-Host "    Report: $StaticReportMd"
} elseif ($Pipeline -eq "all" -or $Pipeline -eq "static") {
    Write-Section "PIPELINE 1 — Static Multi-Label Baseline"
    Write-Host "  [SKIPPED or FAILED — results not available]" -ForegroundColor Red
}

# ------ Pipeline 2a Summary ------
if ($results.ContainsKey("temporal") -and $null -ne $results["temporal"]) {
    $r = $results["temporal"]
    $t = $r.temporal_model_evaluation.test_metrics_optimized_threshold
    $p = $r.prior_baseline_evaluation.test_metrics_optimized_threshold
    $thr = $r.temporal_model_evaluation.val_threshold_optimization.best_threshold
    $obs = $r.dataset_observations
    $ts  = $r.training_summary

    Write-Section "PIPELINE 2a — Temporal Multi-Label Baseline (363 classes)"
    Write-Host "  Dataset:       temporal_multilabel_frequent363_dataset.csv" -ForegroundColor Gray
    Write-Host "  Split:         Train=$($obs.train) / Val=$($obs.val) / Test=$($obs.test)"
    Write-Host "  Features:      9 admission temporal features (age + eMAR counts + deltas)"
    Write-Host "  Best Epoch:    $($ts.best_epoch) / $($ts.total_epochs)  |  Val Loss: $(Format-Metric $ts.best_val_loss 6)"
    Write-Host "  Val Threshold: $thr (optimized on validation set)"
    Write-Host ""
    Write-Host "  Test Results (Temporal LR | val-optimized threshold $thr):" -ForegroundColor White
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f "Model","Micro-F1","Macro-F1","Hamm.Loss","mAP","P@5","R@5")
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f ("="*18),("="*10),("="*10),("="*12),("="*8),("="*8),("="*8))
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f "Temporal LR",(Format-Metric $t.micro_f1),(Format-Metric $t.macro_f1),(Format-Metric $t.hamming_loss),(Format-Metric $t.mAP),(Format-Metric $t.precision_at_5),(Format-Metric $t.recall_at_5)) -ForegroundColor Green
    Write-Host ("  {0,-18} {1,10} {2,10} {3,12} {4,8} {5,8} {6,8}" -f "Training Prior",(Format-Metric $p.micro_f1),(Format-Metric $p.macro_f1),(Format-Metric $p.hamming_loss),(Format-Metric $p.mAP),(Format-Metric $p.precision_at_5),(Format-Metric $p.recall_at_5))
    Write-Host ""
    Write-Host "  Outputs:"
    Write-Host "    JSON:   $TemporalResultsJson"
    Write-Host "    Report: $TemporalReportMd"
} elseif ($Pipeline -eq "all" -or $Pipeline -eq "temporal") {
    Write-Section "PIPELINE 2a — Temporal Multi-Label Baseline"
    Write-Host "  [SKIPPED or FAILED — results not available]" -ForegroundColor Red
}

# ------ Pipeline 2b Summary ------
if ($results.ContainsKey("ablation") -and $null -ne $results["ablation"]) {
    $r = $results["ablation"]
    $cfgs = $r.configurations
    $obs = $r.dataset_split_counts

    Write-Section "PIPELINE 2b — Temporal Feature Ablation Study (363 classes)"
    Write-Host "  Dataset:       temporal_multilabel_frequent363_dataset.csv" -ForegroundColor Gray
    Write-Host "  Split:         Train=$($obs.train) / Val=$($obs.val) / Test=$($obs.test)"
    Write-Host "  Configurations: 4 (Prior, Age-Only, Temporal-Only, Age+Temporal)"
    Write-Host ""
    Write-Host ("  Test Ablation Results (val-optimized threshold):") -ForegroundColor White
    Write-Host ("  {0,-24} {1,5} {2,10} {3,10} {4,8} {5,8}" -f "Configuration","Dim","Micro-F1","Macro-F1","mAP","Threshold")
    Write-Host ("  {0,-24} {1,5} {2,10} {3,10} {4,8} {5,8}" -f ("="*24),("="*5),("="*10),("="*10),("="*8),("="*8))

    $cfgOrder = @("prior_baseline","age_only","temporal_only","age_temporal")
    foreach ($key in $cfgOrder) {
        if ($cfgs.PSObject.Properties.Name -contains $key) {
            $cfg = $cfgs.$key
            $m = $cfg.test_metrics_optimized_threshold
            $thr = $cfg.val_threshold_optimization.best_threshold
            $dim = $cfg.input_dim
            $name = $cfg.config_name
            Write-Host ("  {0,-24} {1,5} {2,10} {3,10} {4,8} {5,8}" -f $name,$dim,(Format-Metric $m.micro_f1),(Format-Metric $m.macro_f1),(Format-Metric $m.mAP),$thr)
        }
    }

    Write-Host ""
    Write-Host "  NOTE: Test set = $($obs.test) observations. Results are observational signals only." -ForegroundColor DarkYellow
    Write-Host ""
    Write-Host "  Outputs:"
    Write-Host "    JSON:   $AblationResultsJson"
    Write-Host "    Report: $AblationReportMd"
} elseif ($Pipeline -eq "all" -or $Pipeline -eq "ablation") {
    Write-Section "PIPELINE 2b — Temporal Feature Ablation"
    Write-Host "  [SKIPPED or FAILED — results not available]" -ForegroundColor Red
}

# ============================================================
# Cross-pipeline comparison (only when both are available)
# ============================================================

$hasStatic   = $results.ContainsKey("static")   -and $null -ne $results["static"]
$hasTemporal = $results.ContainsKey("temporal") -and $null -ne $results["temporal"]

if ($hasStatic -and $hasTemporal) {
    Write-Section "CROSS-PIPELINE COMPARISON (Same 363-class Target)"

    $s_t = $results["static"].model_evaluation.test_metrics_optimized_threshold
    $s_thr = $results["static"].model_evaluation.val_threshold_optimization.best_threshold
    $t_t = $results["temporal"].temporal_model_evaluation.test_metrics_optimized_threshold
    $t_thr = $results["temporal"].temporal_model_evaluation.val_threshold_optimization.best_threshold

    Write-Host ""
    Write-Host ("  {0,-28} {1,5} {2,10} {3,10} {4,12} {5,8} {6,8} {7,8}" -f "Pipeline","Thr","Micro-F1","Macro-F1","Hamm.Loss","mAP","P@5","R@5")
    Write-Host ("  {0,-28} {1,5} {2,10} {3,10} {4,12} {5,8} {6,8} {7,8}" -f ("="*28),("="*5),("="*10),("="*10),("="*12),("="*8),("="*8),("="*8))

    $s_color = if ([double]$s_t.micro_f1 -ge [double]$t_t.micro_f1) { "Green" } else { "White" }
    $t_color = if ([double]$t_t.micro_f1 -gt [double]$s_t.micro_f1) { "Green" } else { "White" }

    Write-Host ("  {0,-28} {1,5} {2,10} {3,10} {4,12} {5,8} {6,8} {7,8}" -f "P1: Static FP LR",$s_thr,(Format-Metric $s_t.micro_f1),(Format-Metric $s_t.macro_f1),(Format-Metric $s_t.hamming_loss),(Format-Metric $s_t.mAP),(Format-Metric $s_t.precision_at_5),(Format-Metric $s_t.recall_at_5)) -ForegroundColor $s_color
    Write-Host ("  {0,-28} {1,5} {2,10} {3,10} {4,12} {5,8} {6,8} {7,8}" -f "P2a: Temporal LR",$t_thr,(Format-Metric $t_t.micro_f1),(Format-Metric $t_t.macro_f1),(Format-Metric $t_t.hamming_loss),(Format-Metric $t_t.mAP),(Format-Metric $t_t.precision_at_5),(Format-Metric $t_t.recall_at_5)) -ForegroundColor $t_color

    Write-Host ""
    Write-Host "  IMPORTANT: Both pipelines use the same 363-class target." -ForegroundColor DarkYellow
    Write-Host "  Pipeline 1 test set: $($results["static"].dataset_observations.test) pairs (static split)" -ForegroundColor DarkYellow
    Write-Host "  Pipeline 2 test set: $($results["temporal"].dataset_observations.test) pairs (temporal split)" -ForegroundColor DarkYellow
    Write-Host "  Direct comparison is valid only if both use identical test pairs." -ForegroundColor DarkYellow
}

# ============================================================
# Final Status
# ============================================================

Write-Header "TRAINING WORKFLOW STATUS"

if ($failures.Count -eq 0) {
    Write-Host "  ALL PIPELINES COMPLETED SUCCESSFULLY" -ForegroundColor Green
} else {
    Write-Host "  FAILURES DETECTED:" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host "    - $f" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  Results directory: $(Join-Path $ProjectRoot 'results\metrics\')"
Write-Host "  Reports directory: $(Join-Path $ProjectRoot 'reports\')"
Write-Host ""
Write-Host "  Tip: Re-run a single pipeline with:"
Write-Host "    .\train_ddi.ps1 -Pipeline static"
Write-Host "    .\train_ddi.ps1 -Pipeline temporal"
Write-Host "    .\train_ddi.ps1 -Pipeline ablation"
Write-Host ""
