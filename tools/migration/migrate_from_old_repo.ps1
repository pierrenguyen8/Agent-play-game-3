param(
    [Parameter(Mandatory=$true)]
    [string]$OldRepo,

    [string]$NewRepo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")),

    [switch]$IncludeRawData,
    [switch]$IncludeProcessedData
)

$ErrorActionPreference = "Stop"

function Copy-IfExists {
    param([string]$Source, [string]$Destination)
    if (Test-Path $Source) {
        $parent = Split-Path $Destination -Parent
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Copy-Item -Recurse -Force $Source $Destination
        Write-Host "COPIED: $Source -> $Destination"
    } else {
        Write-Host "SKIP (missing): $Source"
    }
}

# Selected best weights only — no generated training directories.
Copy-IfExists `
    (Join-Path $OldRepo "runs\detect\training_runs\planet\player_detector\weights\best.pt") `
    (Join-Path $NewRepo "artifacts\planet\perception\player\best.pt")

Copy-IfExists `
    (Join-Path $OldRepo "data\perception\planet\enemy\training_runs\enemy_detector\weights\best.pt") `
    (Join-Path $NewRepo "artifacts\planet\perception\enemy\best.pt")

Copy-IfExists `
    (Join-Path $OldRepo "data\perception\planet\platform\training_runs\platform_detector\best.pt") `
    (Join-Path $NewRepo "artifacts\planet\perception\platform\best.pt")

# Legacy low-level checkpoint is benchmark-only.
Copy-IfExists `
    (Join-Path $OldRepo "data\low_level\planet\training_runs\final_temporal_v1\best.pt") `
    (Join-Path $NewRepo "artifacts\legacy\low_level\final_temporal_v1_best.pt")

Copy-IfExists `
    (Join-Path $OldRepo "data\low_level\planet\training_runs\final_temporal_v1\val_threshold_eval.json") `
    (Join-Path $NewRepo "artifacts\legacy\low_level\val_threshold_eval.json")

if ($IncludeRawData) {
    Copy-IfExists `
        (Join-Path $OldRepo "data\raw") `
        (Join-Path $NewRepo "data\raw")
}

if ($IncludeProcessedData) {
    Copy-IfExists `
        (Join-Path $OldRepo "data\low_level\planet\splits\final_temporal_v1.json") `
        (Join-Path $NewRepo "data\legacy_temporal\split.json")
    Copy-IfExists `
        (Join-Path $OldRepo "data\low_level\planet\temporal_timeline") `
        (Join-Path $NewRepo "data\legacy_temporal\timeline")
    Copy-IfExists `
        (Join-Path $OldRepo "data\low_level\planet\temporal_sequences") `
        (Join-Path $NewRepo "data\legacy_temporal\sequences")
    Copy-IfExists `
        (Join-Path $OldRepo "data\low_level\planet\visual_cache") `
        (Join-Path $NewRepo "data\legacy_temporal\visual_cache")
}

Write-Host ""
Write-Host "Migration finished. No Python source was copied from the old repo."
