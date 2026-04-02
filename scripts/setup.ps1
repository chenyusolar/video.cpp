# video.cpp Environment Setup Script
# This script installs FFmpeg and configures GPU backend support (CUDA/Vulkan)
# Usage: .\setup.ps1

param(
    [switch]$InstallFFmpeg,
    [switch]$InstallCUDA,
    [switch]$InstallVulkan,
    [switch]$InstallAll,
    [switch]$SkipInstallation,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

Write-Host "video.cpp Environment Setup" -ForegroundColor Cyan
Write-Host "=" * 60

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Check if running as admin (needed for some installations)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

# ============================================================
# FFmpeg Installation
# ============================================================
function Install-FFmpeg {
    Write-Host "`n[FFmpeg Setup]" -ForegroundColor Yellow
    
    # Check if already installed
    $ffmpegPath = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegPath) {
        $version = & ffmpeg -version 2>&1 | Select-Object -First 1
        Write-Host "FFmpeg is already installed: $version" -ForegroundColor Green
        return $true
    }
    
    # Check common installation paths
    $commonPaths = @(
        "C:\ffmpeg\bin\ffmpeg.exe",
        "$env:LOCALAPPDATA\Programs\ffmpeg\bin\ffmpeg.exe",
        "$env:ProgramFiles\ffmpeg\bin\ffmpeg.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            # Add to PATH if not already there
            $binDir = Split-Path $path -Parent
            if ($env:PATH -notlike "*$binDir*") {
                $env:PATH = "$binDir;$env:PATH"
                Write-Host "Added $binDir to PATH" -ForegroundColor Green
            }
            Write-Host "FFmpeg found at: $path" -ForegroundColor Green
            return $true
        }
    }
    
    if ($SkipInstallation) {
        Write-Host "FFmpeg not found. Skipping installation." -ForegroundColor Yellow
        return $false
    }
    
    Write-Host "FFmpeg not found. Installing..." -ForegroundColor Yellow
    
    if (-not $isAdmin) {
        Write-Host "Administrator privileges recommended for FFmpeg installation." -ForegroundColor Yellow
    }
    
    # Download and install FFmpeg
    $ffmpegVersion = "7.1"
    $ffmpegDir = "$env:LOCALAPPDATA\Programs\ffmpeg"
    $ffmpegZip = "$env:TEMP\ffmpeg.zip"
    $ffmpegUrl = "https://github.com/GyanD/codexffmpeg/releases/download/$ffmpegVersion/ffmpeg-$ffmpegVersion-full_build.zip"
    
    Write-Host "Downloading FFmpeg $ffmpegVersion..." -ForegroundColor Cyan
    
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $ffmpegUrl -OutFile $ffmpegZip -UseBasicParsing
        
        Write-Host "Extracting..." -ForegroundColor Cyan
        Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegDir -Force
        
        # Find the actual extracted folder
        $extractedDir = Get-ChildItem -Path $ffmpegDir -Directory | Where-Object { $_.Name -like "ffmpeg*" } | Select-Object -First 1
        
        if ($extractedDir) {
            $binPath = "$($extractedDir.FullName)\bin"
            
            # Add to system PATH (requires admin)
            if ($isAdmin) {
                $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
                if ($userPath -notlike "*$binPath*") {
                    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$binPath", "User")
                    $env:PATH = "$binPath;$env:PATH"
                }
                Write-Host "FFmpeg installed and added to system PATH" -ForegroundColor Green
            } else {
                # Add to current session PATH only
                $env:PATH = "$binPath;$env:PATH"
                Write-Host "FFmpeg installed to: $binPath" -ForegroundColor Green
                Write-Host "Note: Run as Administrator to add FFmpeg to system PATH permanently" -ForegroundColor Yellow
            }
        }
        
        # Cleanup
        Remove-Item $ffmpegZip -Force -ErrorAction SilentlyContinue
        
        Write-Host "FFmpeg installed successfully!" -ForegroundColor Green
        return $true
        
    } catch {
        Write-Host "Failed to install FFmpeg: $_" -ForegroundColor Red
        Write-Host "Please install manually from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
        return $false
    }
}

# ============================================================
# CUDA Installation Check and Setup
# ============================================================
function Test-CUDAInstalled {
    $cudaPath = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($cudaPath) {
        $version = & nvcc --version 2>&1 | Select-Object -Last 1
        Write-Host "CUDA is installed: $version" -ForegroundColor Green
        return $true
    }
    
    # Check NVIDIA driver
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        Write-Host "NVIDIA driver found" -ForegroundColor Green
        
        # Get GPU info
        try {
            $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
            Write-Host "GPU: $gpuInfo" -ForegroundColor Cyan
        } catch {
            # nvidia-smi might not be in PATH
        }
    }
    
    return $false
}

function Install-CUDA {
    Write-Host "`n[CUDA Setup]" -ForegroundColor Yellow
    
    if (Test-CUDAInstalled) {
        Write-Host "CUDA toolkit is already configured." -ForegroundColor Green
        
        # Check Rust CUDA support
        $cudaVersion = & nvcc --version 2>&1 | Select-Object -Last 1
        Write-Host "CUDA Version: $cudaVersion" -ForegroundColor Cyan
        
        return $true
    }
    
    if ($SkipInstallation) {
        Write-Host "CUDA not found. Skipping installation." -ForegroundColor Yellow
        Write-Host "To install CUDA, visit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
        return $false
    }
    
    Write-Host "CUDA toolkit not found." -ForegroundColor Yellow
    Write-Host "`nTo install CUDA:" -ForegroundColor Cyan
    Write-Host "  1. Visit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    Write-Host "  2. Download CUDA Toolkit (11.8 or later recommended)" -ForegroundColor White
    Write-Host "  3. Run the installer" -ForegroundColor White
    Write-Host "  4. Restart your terminal and run this script again" -ForegroundColor White
    Write-Host "`nAfter installation, ensure nvcc is in your PATH." -ForegroundColor Yellow
    
    # Try to open the download page
    try {
        Start-Process "https://developer.nvidia.com/cuda-downloads"
        Write-Host "`nOpened CUDA download page in browser." -ForegroundColor Green
    } catch {
        # Silently fail if browser opening fails
    }
    
    return $false
}

# ============================================================
# Vulkan Installation Check and Setup
# ============================================================
function Test-VulkanInstalled {
    $vulkanInfo = & vulkaninfo --version 2>&1
    if ($LASTEXITCODE -eq 0 -and $vulkanInfo) {
        Write-Host "Vulkan is installed: $vulkanInfo" -ForegroundColor Green
        return $true
    }
    
    # Check for Vulkan runtime
    $vulkanRuntime = Get-ItemProperty -Path "HKLM:\SOFTWARE\Khronos\Vulkan\Drivers" -ErrorAction SilentlyContinue
    if ($vulkanRuntime) {
        Write-Host "Vulkan registry entries found" -ForegroundColor Green
        return $true
    }
    
    return $false
}

function Install-Vulkan {
    Write-Host "`n[Vulkan Setup]" -ForegroundColor Yellow
    
    if (Test-VulkanInstalled) {
        Write-Host "Vulkan runtime is already configured." -ForegroundColor Green
        return $true
    }
    
    if ($SkipInstallation) {
        Write-Host "Vulkan not found. Skipping installation." -ForegroundColor Yellow
        Write-Host "To install Vulkan, visit: https://vulkan.lunarg.com/" -ForegroundColor Cyan
        return $false
    }
    
    Write-Host "Vulkan runtime not found." -ForegroundColor Yellow
    Write-Host "`nTo install Vulkan SDK:" -ForegroundColor Cyan
    Write-Host "  1. Visit: https://vulkan.lunarg.com/sdk/home" -ForegroundColor White
    Write-Host "  2. Download the Windows SDK installer" -ForegroundColor White
    Write-Host "  3. Run the installer" -ForegroundColor White
    Write-Host "  4. Restart your terminal and run this script again" -ForegroundColor White
    
    # Try to open the download page
    try {
        Start-Process "https://vulkan.lunarg.com/sdk/home"
        Write-Host "`nOpened Vulkan SDK download page in browser." -ForegroundColor Green
    } catch {
        # Silently fail if browser opening fails
    }
    
    return $false
}

# ============================================================
# GPU Backend Compilation Check
# ============================================================
function Test-GPUBackendCompilation {
    param(
        [string]$Backend
    )
    
    Write-Host "`n[Checking $Backend Backend Compilation]" -ForegroundColor Yellow
    
    $features = ""
    if ($Backend -eq "cuda") {
        $features = "--features cuda"
    } elseif ($Backend -eq "vulkan") {
        $features = "--features vulkan"
    }
    
    Push-Location "$ProjectRoot\core"
    $output = & cargo check $features 2>&1
    Pop-Location
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$Backend backend compiles successfully!" -ForegroundColor Green
        return $true
    } else {
        # Check if it's a feature not enabled error
        if ($output -like "*feature*not*found*" -or $output -like "*Invalid*feature*") {
            Write-Host "$Backend feature not available in this build configuration." -ForegroundColor Yellow
            return $false
        }
        
        Write-Host "Compilation issues detected:" -ForegroundColor Yellow
        $output | Select-Object -First 10
        return $false
    }
}

# ============================================================
# Main Setup Logic
# ============================================================

# If CheckOnly, just check everything and exit
if ($CheckOnly) {
    Write-Host "`n=== Environment Check ===" -ForegroundColor Cyan
    
    $ffmpeg = Install-FFmpeg
    $cuda = Test-CUDAInstalled
    $vulkan = Test-VulkanInstalled
    
    Write-Host "`n=== Summary ===" -ForegroundColor Cyan
    Write-Host "FFmpeg: $(if ($ffmpeg) { 'OK' } else { 'NOT FOUND' })" -ForegroundColor $(if ($ffmpeg) { 'Green' } else { 'Red' })
    Write-Host "CUDA: $(if ($cuda) { 'OK' } else { 'NOT FOUND' })" -ForegroundColor $(if ($cuda) { 'Green' } else { 'Red' })
    Write-Host "Vulkan: $(if ($vulkan) { 'OK' } else { 'NOT FOUND' })" -ForegroundColor $(if ($vulkan) { 'Green' } else { 'Red' })
    
    exit 0
}

# Install all if requested
if ($InstallAll) {
    $InstallFFmpeg = $true
    $InstallCUDA = $true
    $InstallVulkan = $true
}

# Run installations
$ffmpegInstalled = $false
$cudaInstalled = $false
$vulkanInstalled = $false

if ($InstallFFmpeg -or $InstallAll) {
    $ffmpegInstalled = Install-FFmpeg
}

if ($InstallCUDA -or $InstallAll) {
    $cudaInstalled = Install-CUDA
}

if ($InstallVulkan -or $InstallAll) {
    $vulkanInstalled = Install-Vulkan
}

# ============================================================
# Build with Backend Support
# ============================================================
Write-Host "`n=== Building with Backend Support ===" -ForegroundColor Cyan

# Check GPU backends
$cudaAvailable = $false
$vulkanAvailable = $false

# Try to compile with CUDA
Write-Host "`nChecking CUDA backend compilation..." -ForegroundColor Yellow
Push-Location "$ProjectRoot\core"
$cargoOutput = & cargo check --features cuda 2>&1
if ($LASTEXITCODE -eq 0) {
    $cudaAvailable = $true
    Write-Host "CUDA backend: COMPILABLE" -ForegroundColor Green
} else {
    if ($cargoOutput -like "*feature*not*found*" -or $cargoOutput -like "*Invalid*feature*") {
        Write-Host "CUDA backend: Feature not available in this build" -ForegroundColor Yellow
    } else {
        Write-Host "CUDA backend: Compilation has issues (may need CUDA toolkit)" -ForegroundColor Yellow
    }
}
Pop-Location

# Try to compile with Vulkan
Write-Host "`nChecking Vulkan backend compilation..." -ForegroundColor Yellow
Push-Location "$ProjectRoot\core"
$vulkanOutput = & cargo check --features vulkan 2>&1
if ($LASTEXITCODE -eq 0) {
    $vulkanAvailable = $true
    Write-Host "Vulkan backend: COMPILABLE" -ForegroundColor Green
} else {
    if ($vulkanOutput -like "*feature*not*found*" -or $vulkanOutput -like "*Invalid*feature*") {
        Write-Host "Vulkan backend: Feature not available in this build" -ForegroundColor Yellow
    } else {
        Write-Host "Vulkan backend: Compilation has issues" -ForegroundColor Yellow
    }
}
Pop-Location

# ============================================================
# Final Summary and Instructions
# ============================================================
Write-Host "`n" -ForegroundColor White
Write-Host "=" * 60
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60

Write-Host "`n[Installation Status]" -ForegroundColor Cyan
Write-Host "FFmpeg: $(if ($ffmpegInstalled) { 'Installed' } else { 'Not installed' })" -ForegroundColor $(if ($ffmpegInstalled) { 'Green' } else { 'Yellow' })
Write-Host "CUDA Toolkit: $(if ($cudaInstalled) { 'Installed' } else { 'Not installed' })" -ForegroundColor $(if ($cudaInstalled) { 'Green' } else { 'Yellow' })
Write-Host "Vulkan SDK: $(if ($vulkanInstalled) { 'Installed' } else { 'Not installed' })" -ForegroundColor $(if ($vulkanInstalled) { 'Green' } else { 'Yellow' })

Write-Host "`n[Backend Compilation Status]" -ForegroundColor Cyan
Write-Host "CUDA Backend: $(if ($cudaAvailable) { 'Ready' } else { 'Not ready' })" -ForegroundColor $(if ($cudaAvailable) { 'Green' } else { 'Yellow' })
Write-Host "Vulkan Backend: $(if ($vulkanAvailable) { 'Ready' } else { 'Not ready' })" -ForegroundColor $(if ($vulkanAvailable) { 'Green' } else { 'Yellow' })

Write-Host "`n[Next Steps]" -ForegroundColor Cyan

if (-not $ffmpegInstalled) {
    Write-Host "1. Install FFmpeg for video encoding support" -ForegroundColor Yellow
    Write-Host "   Run: .\setup.ps1 -InstallFFmpeg" -ForegroundColor White
}

if (-not $cudaInstalled -and -not $vulkanInstalled) {
    Write-Host "2. Install GPU backend for acceleration:" -ForegroundColor Yellow
    Write-Host "   - CUDA: Visit https://developer.nvidia.com/cuda-downloads" -ForegroundColor White
    Write-Host "   - Vulkan: Visit https://vulkan.lunarg.com/sdk/home" -ForegroundColor White
}

if ($cudaAvailable -or $vulkanAvailable) {
    Write-Host "`nTo build with GPU backend:" -ForegroundColor Green
    if ($cudaAvailable) {
        Write-Host "   Release build: .\build.ps1 -Release -WithCUDA" -ForegroundColor White
    }
    if ($vulkanAvailable) {
        Write-Host "   Vulkan build: .\build.ps1 -Release -WithVulkan" -ForegroundColor White
    }
}

Write-Host "`nTo use CPU backend (no GPU needed):" -ForegroundColor Cyan
Write-Host "   Release build: .\build.ps1 -Release" -ForegroundColor White
Write-Host "   Note: CPU inference is slower for large models" -ForegroundColor Gray

Write-Host "`n[Environment Variables]" -ForegroundColor Cyan
Write-Host "Set VIDEO_MODEL_PATH to your GGUF model file:" -ForegroundColor White
Write-Host '   $env:VIDEO_MODEL_PATH = "models\ltx-2.3-22b-dev-Q4_K_M.gguf"' -ForegroundColor Gray

Write-Host "`n" -ForegroundColor White
