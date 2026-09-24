param(
    [string]$Destination = "",
    [string]$Ref = "main"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Destination) {
    $Destination = Join-Path $repoRoot "external\pixelbc"
}

$parent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $parent | Out-Null

if (Test-Path $Destination) {
    throw "Destination already exists: $Destination"
}

$upstream = "https://github.com/microsoft/imitation_learning_in_modern_video_games.git"
Write-Host "Cloning PixelBC upstream source only (not installing dependencies)..."
git clone --depth 1 --branch $Ref $upstream $Destination

$commit = (git -C $Destination rev-parse HEAD).Trim()
$record = Join-Path $repoRoot "external\pixelbc.commit.txt"
@(
    "repository=$upstream",
    "ref=$Ref",
    "commit=$commit",
    "license=MIT"
) | Set-Content -Encoding UTF8 $record

Write-Host "PixelBC source: $Destination"
Write-Host "Pinned record: $record"
Write-Host "Do NOT pip install the upstream package into the Planet runtime venv."
