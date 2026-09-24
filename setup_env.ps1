param(
    [switch]$WithML,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot ".venv"

function Test-PythonCandidate {
    param(
        [string]$Command,
        [string[]]$PrefixArgs
    )

    try {
        $versionText = & $Command @PrefixArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $versionText) {
            return $null
        }

        $parts = $versionText.Trim().Split('.')
        $major = [int]$parts[0]
        $minor = [int]$parts[1]

        # Keep the bootstrap reproducible for the current project.
        # Python 3.11 and 3.12 are supported here.
        if ($major -eq 3 -and ($minor -eq 11 -or $minor -eq 12)) {
            return [PSCustomObject]@{
                Command = $Command
                PrefixArgs = $PrefixArgs
                Version = $versionText.Trim()
            }
        }
    }
    catch {
        return $null
    }

    return $null
}

function Find-CompatiblePython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($selector in @("-3.12", "-3.11")) {
            $candidate = Test-PythonCandidate -Command "py" -PrefixArgs @($selector)
            if ($null -ne $candidate) {
                return $candidate
            }
        }
    }

    foreach ($command in @("python", "python3")) {
        if (Get-Command $command -ErrorAction SilentlyContinue) {
            $candidate = Test-PythonCandidate -Command $command -PrefixArgs @()
            if ($null -ne $candidate) {
                return $candidate
            }
        }
    }

    return $null
}

if ($Recreate -and (Test-Path $VenvPath)) {
    Write-Host "Removing existing .venv..."
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPath)) {
    $PythonCandidate = Find-CompatiblePython

    if ($null -eq $PythonCandidate) {
        Write-Host ""
        Write-Host "No compatible Python interpreter was found." -ForegroundColor Yellow
        Write-Host "This project currently supports Python 3.11 or 3.12." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Installed Python launchers (if any):"
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -0p
        }
        if (Get-Command python -ErrorAction SilentlyContinue) {
            & python --version
        }
        Write-Host ""
        throw "Install Python 3.11 or 3.12, then run setup_env.cmd again."
    }

    Write-Host "Using Python $($PythonCandidate.Version)"
    Write-Host "Creating virtual environment at .venv..."
    & $PythonCandidate.Command @($PythonCandidate.PrefixArgs) -m venv .venv

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv with Python $($PythonCandidate.Version)."
    }
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found at $Python"
}

$VenvVersion = & $Python -c "import sys; print(sys.version.split()[0])"
Write-Host "Virtual environment Python: $VenvVersion"

Write-Host "Upgrading packaging tools..."
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip/setuptools/wheel."
}

$Extras = "dev,planet-runtime"
if ($WithML) {
    $Extras = "dev,planet-runtime,planet-perception,ml"
}

Write-Host "Installing Personal Agent [$Extras]..."
& $Python -m pip install -e ".[${Extras}]"
if ($LASTEXITCODE -ne 0) {
    throw "Project dependency installation failed."
}

Write-Host "Running tests..."
& $Python -m pytest
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed after environment setup."
}

Write-Host ""
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate in PowerShell with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Activation is optional. You can always run:"
Write-Host "  .\.venv\Scripts\python.exe <command>"

if (-not $WithML) {
    Write-Host ""
    Write-Host "Perception/ML dependencies were not installed."
    Write-Host "When needed, run:"
    Write-Host "  .\setup_env.cmd -WithML"
}
