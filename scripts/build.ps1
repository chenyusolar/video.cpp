# Build script for video.cpp
# Usage: .\build.ps1

param(
    [switch]$Release,
    [switch]$WithCUDA,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "video.cpp Build Script" -ForegroundColor Cyan
Write-Host "=" * 50

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Clean if requested
if ($Clean) {
    Write-Host "`nCleaning build artifacts..." -ForegroundColor Yellow
    if (Test-Path "$ProjectRoot\core\target") {
        Remove-Item -Recurse -Force "$ProjectRoot\core\target"
    }
    if (Test-Path "$ProjectRoot\bin") {
        Remove-Item -Recurse -Force "$ProjectRoot\bin"
    }
    Write-Host "Clean complete." -ForegroundColor Green
}

# Build Rust core
Write-Host "`n[1/3] Building Rust core..." -ForegroundColor Yellow

$Features = ""
if ($WithCUDA) {
    $Features = "--features cuda"
    Write-Host "Building with CUDA support..." -ForegroundColor Cyan
}

$cargoArgs = @("build")
if ($Release) {
    $cargoArgs += "--release"
}
if ($Features) {
    $cargoArgs += $Features.Split(" ")
}

Push-Location "$ProjectRoot\core"
& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Rust build failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location
Write-Host "Rust core built successfully." -ForegroundColor Green

# Build CLI
Write-Host "`n[2/3] Building Go CLI..." -ForegroundColor Yellow

# Create bin directory if not exists
if (-not (Test-Path "$ProjectRoot\bin")) {
    New-Item -ItemType Directory -Path "$ProjectRoot\bin" | Out-Null
}

Push-Location "$ProjectRoot\cli"
$goArgs = @("build")
if ($Release) {
    $goArgs += "-ldflags=-s -w"
}
$goArgs += "-o", "$ProjectRoot\bin\video.exe"

& go @goArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Go build failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location
Write-Host "CLI built successfully." -ForegroundColor Green

Write-Host "`n[3/3] Build complete!" -ForegroundColor Green
Write-Host "=" * 50

Write-Host "`nOutputs:" -ForegroundColor Cyan
Write-Host "  CLI: $ProjectRoot\bin\video.exe"
Write-Host "  Core: $ProjectRoot\core\target\release\video_core.dll"

Write-Host "`nTo run:" -ForegroundColor Yellow
Write-Host '  $env:VIDEO_MODEL_PATH = "models\ltx-2.3-22b-dev-Q4_K_M.gguf"'
Write-Host "  .\bin\video.exe -p `"a dragon flying`""
