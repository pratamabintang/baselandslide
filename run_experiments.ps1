<#
.SYNOPSIS
    Automated Sequential Training Script for Landslide Multi-Modal Segmentation Experiments.

.DESCRIPTION
    Sequentially trains and benchmarks combinations of:
      - Modalities: RGB-only vs. RGB-DTM
      - Architectures: Pure Direct U-Net (No projection) vs. U-Net Proj (1x1 Conv projection)
      - Fusion Modes: Concatenation (concat) vs. Element-wise Addition (add)

.PARAMETER Epochs
    Number of training epochs per experiment (Default: 50).

.PARAMETER BatchSize
    Batch size for training and validation (Default: 8).

.PARAMETER Weights
    Initial backbone weights (e.g. 'unet_carvana' or '' for scratch) (Default: 'unet_carvana').

.PARAMETER Device
    Compute device: '0' for CUDA GPU, 'cpu' for CPU (Default: '0').

.PARAMETER Workers
    Number of DataLoader workers (Default: 2).

.PARAMETER CacheRam
    Switch flag to cache preprocessed arrays in RAM (Default: false).

.PARAMETER Project
    Output parent directory for run artifacts (Default: 'runs/train').

.PARAMETER SelectIds
    Array of specific experiment IDs to run (e.g. -SelectIds 1,3,6). Default runs all 6.

.PARAMETER DryRun
    Preview the experiment queue and command lines without launching training.

.EXAMPLE
    .\run_experiments.ps1
    .\run_experiments.ps1 -Epochs 30 -BatchSize 8
    .\run_experiments.ps1 -SelectIds 1,3,6 -Epochs 50
    .\run_experiments.ps1 -DryRun
#>

[CmdletBinding()]
param(
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [string]$Weights = "unet_carvana",
    [string]$Device = "0",
    [int]$Workers = 2,
    [switch]$CacheRam = $true,
    [string]$Project = "runs/train",
    [object[]]$SelectIds = @(),
    [switch]$DryRun = $false
)

# Ensure UTF-8 output encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ==============================================================================
# 1. Experiment Matrix Definition
# ==============================================================================
$ExperimentMatrix = @(
    [PSCustomObject]@{
        Id          = 1
        Name        = "exp_1_rgb_pure_unet"
        Inputs      = "rgb_only"
        Cfg         = "models/architectures/unet_noproj.yaml"
        Fusion      = "concat"
        ArchName    = "Pure U-Net (No Proj)"
        Description = "RGB-Only (3ch) with Direct U-Net"
    },
    [PSCustomObject]@{
        Id          = 2
        Name        = "exp_2_rgb_unet_proj"
        Inputs      = "rgb_only"
        Cfg         = "models/architectures/unet.yaml"
        Fusion      = "concat"
        ArchName    = "U-Net Proj (1x1 Conv)"
        Description = "RGB-Only (3ch) with 1x1 Conv Projection"
    },
    [PSCustomObject]@{
        Id          = 3
        Name        = "exp_3_rgb_dtm_pure_unet_add"
        Inputs      = "rgb_dtm"
        Cfg         = "models/architectures/unet_noproj.yaml"
        Fusion      = "add"
        ArchName    = "Pure U-Net (No Proj)"
        Description = "RGB-DTM Addition Fusion (3ch) with Direct U-Net"
    },
    [PSCustomObject]@{
        Id          = 4
        Name        = "exp_4_rgb_dtm_pure_unet_concat"
        Inputs      = "rgb_dtm"
        Cfg         = "models/architectures/unet_noproj.yaml"
        Fusion      = "concat"
        ArchName    = "Pure U-Net (No Proj)"
        Description = "RGB-DTM Concat Fusion (4ch) with Direct U-Net"
    },
    [PSCustomObject]@{
        Id          = 5
        Name        = "exp_5_rgb_dtm_unet_proj_add"
        Inputs      = "rgb_dtm"
        Cfg         = "models/architectures/unet.yaml"
        Fusion      = "add"
        ArchName    = "U-Net Proj (1x1 Conv)"
        Description = "RGB-DTM Addition Fusion (3ch) with 1x1 Conv Projection"
    },
    [PSCustomObject]@{
        Id          = 6
        Name        = "exp_6_rgb_dtm_unet_proj_concat"
        Inputs      = "rgb_dtm"
        Cfg         = "models/architectures/unet.yaml"
        Fusion      = "concat"
        ArchName    = "U-Net Proj (1x1 Conv)"
        Description = "RGB-DTM Concat Fusion (4ch) with 1x1 Conv Projection"
    }
)

# Parse and filter experiments if SelectIds specified
$parsedIds = @()
if ($SelectIds -and $SelectIds.Count -gt 0) {
    foreach ($item in $SelectIds) {
        $strItem = "$item"
        if ($strItem.Contains(',')) {
            $parsedIds += $strItem.Split(',') | ForEach-Object { if ($_ -match '\d+') { [int]$_.Trim() } }
        } elseif ($strItem -match '\d+') {
            $parsedIds += [int]$strItem.Trim()
        }
    }
}

$ActiveQueue = @()
if ($parsedIds.Count -gt 0) {
    foreach ($exp in $ExperimentMatrix) {
        if ($parsedIds -contains [int]$exp.Id) {
            $ActiveQueue += $exp
        }
    }
} else {
    $ActiveQueue = $ExperimentMatrix
}

# ==============================================================================
# 2. Header and Queue Overview
# ==============================================================================
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SEQUENTIAL MULTI-MODAL LANDSLIDE TRAINING SUITE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Total Experiments : $($ActiveQueue.Count) in queue"
Write-Host "  Epochs per Run    : $Epochs"
Write-Host "  Batch Size        : $BatchSize"
Write-Host "  Pretrained Weights: $Weights"
Write-Host "  Compute Device    : $Device"
Write-Host "  Workers           : $Workers"
Write-Host "  RAM Caching       : $(if ($CacheRam) { 'Enabled' } else { 'Disabled (Safe)' })"
Write-Host "  Output Directory  : $Project"
Write-Host "--------------------------------------------------------------------------------"
Write-Host "  EXPERIMENT QUEUE:" -ForegroundColor Yellow

foreach ($exp in $ActiveQueue) {
    Write-Host ("   [{0}] {1,-30} | Inputs: {2,-8} | Arch: {3,-20} | Fusion: {4}" -f $exp.Id, $exp.Name, $exp.Inputs, $exp.ArchName, $exp.Fusion)
}
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] Generated CLI Commands:" -ForegroundColor Yellow
    foreach ($exp in $ActiveQueue) {
        $cacheArg = if ($CacheRam) { "--cache-ram" } else { "" }
        $cmd = "python train.py --cfg $($exp.Cfg) --inputs $($exp.Inputs) --fusion $($exp.Fusion) --weights $Weights --epochs $Epochs --batch-size $BatchSize --device $Device --workers $Workers --project $Project --name $($exp.Name) $cacheArg"
        Write-Host "  $cmd" -ForegroundColor Green
    }
    Write-Host "`n[DRY RUN] Finished without executing." -ForegroundColor Yellow
    exit 0
}

# ==============================================================================
# 3. Execution Loop
# ==============================================================================
$SuiteStartTime = Get-Date
$ResultsList = @()
$TotalCount = $ActiveQueue.Count
$CurrentIndex = 0

foreach ($exp in $ActiveQueue) {
    $CurrentIndex++
    $RunStartTime = Get-Date
    
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Green
    Write-Host ("[{0}/{1}] STARTING: {2}" -f $CurrentIndex, $TotalCount, $exp.Name) -ForegroundColor Green
    Write-Host ("  Description  : {0}" -f $exp.Description)
    Write-Host ("  Inputs       : {0} (Fusion: {1})" -f $exp.Inputs, $exp.Fusion)
    Write-Host ("  Architecture : {0}" -f $exp.Cfg)
    Write-Host ("  Start Time   : {0}" -f $RunStartTime.ToString("yyyy-MM-dd HH:mm:ss"))
    Write-Host ("=" * 80) -ForegroundColor Green
    Write-Host ""

    # Build argument list
    $argList = @(
        "train.py",
        "--cfg", $exp.Cfg,
        "--inputs", $exp.Inputs,
        "--fusion", $exp.Fusion,
        "--weights", $Weights,
        "--epochs", $Epochs.ToString(),
        "--batch-size", $BatchSize.ToString(),
        "--device", $Device,
        "--workers", $Workers.ToString(),
        "--project", $Project,
        "--name", $exp.Name
    )

    if ($CacheRam) {
        $argList += "--cache-ram"
    }

    # Execute training via python
    $process = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -Wait -PassThru
    $exitCode = $process.ExitCode
    $RunEndTime = Get-Date
    $DurationMinutes = [Math]::Round(($RunEndTime - $RunStartTime).TotalMinutes, 2)

    # Parse results from the experiment directory
    $ExpDir = Join-Path $Project $exp.Name
    $BestMetricsFile = Join-Path $ExpDir "best_metrics.yaml"
    $ResultsCsv = Join-Path $ExpDir "results.csv"

    $BestDice = 0.0
    $BestIoU = 0.0
    $BestRecall = 0.0
    $BestPrecision = 0.0
    $ValLoss = 0.0
    $Status = if ($exitCode -eq 0) { "Success" } else { "Failed" }

    if (Test-Path $BestMetricsFile) {
        try {
            $yamlContent = Get-Content $BestMetricsFile -Raw
            # Simple regex parser for YAML metrics
            if ($yamlContent -match 'best_dice:\s*([0-9\.]+)') { $BestDice = [double]$matches[1] }
            if ($yamlContent -match 'best_iou:\s*([0-9\.]+)') { $BestIoU = [double]$matches[1] }
            if ($yamlContent -match 'best_recall:\s*([0-9\.]+)') { $BestRecall = [double]$matches[1] }
            if ($yamlContent -match 'best_precision:\s*([0-9\.]+)') { $BestPrecision = [double]$matches[1] }
            if ($yamlContent -match 'val_loss:\s*([0-9\.]+)') { $ValLoss = [double]$matches[1] }
        } catch {
            Write-Warning "Could not parse $BestMetricsFile"
        }
    }

    # Record experiment result
    $resultRecord = [PSCustomObject]@{
        Id              = $exp.Id
        Name            = $exp.Name
        Inputs          = $exp.Inputs
        Architecture    = $exp.ArchName
        Fusion          = $exp.Fusion
        Status          = $Status
        DurationMin     = $DurationMinutes
        BestDice        = $BestDice
        BestIoU         = $BestIoU
        BestRecall      = $BestRecall
        BestPrecision   = $BestPrecision
        ValLoss         = $ValLoss
    }
    $ResultsList += $resultRecord

    Write-Host ""
    Write-Host ("-" * 80) -ForegroundColor Yellow
    Write-Host ("[{0}/{1}] COMPLETED: {2} in {3} min (Status: {4})" -f $CurrentIndex, $TotalCount, $exp.Name, $DurationMinutes, $Status) -ForegroundColor Yellow
    if ($Status -eq "Success") {
        Write-Host ("  -> Best Val Dice (F1) : {0:P2}" -f $BestDice) -ForegroundColor Green
        Write-Host ("  -> Best Val mIoU      : {0:P2}" -f $BestIoU) -ForegroundColor Green
        Write-Host ("  -> Best Val Recall    : {0:P2}" -f $BestRecall) -ForegroundColor Green
    }
    Write-Host ("-" * 80) -ForegroundColor Yellow
}

# ==============================================================================
# 4. Final Comparison Report & CSV Summary
# ==============================================================================
$SuiteEndTime = Get-Date
$TotalSuiteDurationHours = [Math]::Round(($SuiteEndTime - $SuiteStartTime).TotalHours, 2)

Write-Host ""
Write-Host "==========================================================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK SUMMARY & LEADERBOARD (Total Duration: $TotalSuiteDurationHours hours)" -ForegroundColor Cyan
Write-Host "==========================================================================================" -ForegroundColor Cyan

# Output formatted table to console
$ResultsList | Format-Table -Property @(
    @{Label="ID"; Expression={$_.Id}; Width=4},
    @{Label="Experiment"; Expression={$_.Name}; Width=30},
    @{Label="Inputs"; Expression={$_.Inputs}; Width=10},
    @{Label="Fusion"; Expression={$_.Fusion}; Width=8},
    @{Label="Status"; Expression={$_.Status}; Width=9},
    @{Label="Dice (F1)"; Expression={"{0:P2}" -f $_.BestDice}; Width=11},
    @{Label="mIoU"; Expression={"{0:P2}" -f $_.BestIoU}; Width=10},
    @{Label="Recall"; Expression={"{0:P2}" -f $_.BestRecall}; Width=10},
    @{Label="Time(m)"; Expression={$_.DurationMin}; Width=8}
)

# Export summary to CSV and JSON in project directory
$SummaryCsv = Join-Path $Project "experiments_summary.csv"
$SummaryJson = Join-Path $Project "experiments_summary.json"

try {
    $ResultsList | Export-Csv -Path $SummaryCsv -NoTypeInformation -Encoding UTF8
    $ResultsList | ConvertTo-Json -Depth 3 | Set-Content -Path $SummaryJson -Encoding UTF8
    Write-Host "  Summary CSV Saved  : $SummaryCsv" -ForegroundColor Green
    Write-Host "  Summary JSON Saved : $SummaryJson" -ForegroundColor Green
} catch {
    Write-Warning "Failed to save summary files: $_"
}

Write-Host "==========================================================================================" -ForegroundColor Cyan
Write-Host ""
