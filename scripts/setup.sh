# video.cpp Environment Setup Script (Bash/Linux/macOS)
# This script installs FFmpeg and configures GPU backend support (CUDA/Vulkan)
# Usage: ./setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}video.cpp Environment Setup${NC}"
echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"

# Parse arguments
INSTALL_FFMPEG=false
INSTALL_CUDA=false
INSTALL_VULKAN=false
INSTALL_ALL=false
CHECK_ONLY=false
SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-ffmpeg)
            INSTALL_FFMPEG=true
            shift
            ;;
        --install-cuda)
            INSTALL_CUDA=true
            shift
            ;;
        --install-vulkan)
            INSTALL_VULKAN=true
            shift
            ;;
        --install-all)
            INSTALL_ALL=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================
# FFmpeg Installation
# ============================================================
install_ffmpeg() {
    echo -e "\n${YELLOW}[FFmpeg Setup]${NC}"
    
    # Check if already installed
    if command -v ffmpeg &> /dev/null; then
        VERSION=$(ffmpeg -version 2>&1 | head -1)
        echo -e "${GREEN}FFmpeg is already installed: $VERSION${NC}"
        return 0
    fi
    
    if [ "$SKIP_INSTALL" = true ]; then
        echo -e "${YELLOW}FFmpeg not found. Skipping installation.${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}FFmpeg not found. Installing...${NC}"
    
    # Determine OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use Homebrew
        if command -v brew &> /dev/null; then
            echo "Installing FFmpeg via Homebrew..."
            brew install ffmpeg
        else
            echo -e "${RED}Homebrew not found. Please install FFmpeg manually.${NC}"
            return 1
        fi
    else
        # Linux - use package manager
        if command -v apt-get &> /dev/null; then
            echo "Installing FFmpeg via apt..."
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            echo "Installing FFmpeg via dnf..."
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            echo "Installing FFmpeg via pacman..."
            sudo pacman -S --noconfirm ffmpeg
        else
            echo -e "${RED}No supported package manager found. Please install FFmpeg manually.${NC}"
            echo "Visit: https://ffmpeg.org/download.html"
            return 1
        fi
    fi
    
    echo -e "${GREEN}FFmpeg installed successfully!${NC}"
    return 0
}

# ============================================================
# CUDA Installation Check and Setup
# ============================================================
test_cuda_installed() {
    if command -v nvcc &> /dev/null; then
        VERSION=$(nvcc --version 2>&1 | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo -e "${GREEN}CUDA is installed. Version: $VERSION${NC}"
        
        # Check GPU
        if command -v nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable")
            echo -e "${CYAN}GPU: $GPU_INFO${NC}"
        fi
        return 0
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA driver found.${NC}"
        return 0
    fi
    
    return 1
}

install_cuda() {
    echo -e "\n${YELLOW}[CUDA Setup]${NC}"
    
    if test_cuda_installed; then
        echo -e "${GREEN}CUDA toolkit is already configured.${NC}"
        return 0
    fi
    
    if [ "$SKIP_INSTALL" = true ]; then
        echo -e "${YELLOW}CUDA not found. Skipping installation.${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}CUDA toolkit not found.${NC}"
    echo -e "\n${CYAN}To install CUDA:${NC}"
    echo -e "  1. Visit: https://developer.nvidia.com/cuda-downloads"
    echo -e "  2. Download CUDA Toolkit (11.8 or later recommended)"
    echo -e "  3. Run the installer"
    echo -e "  4. Restart your terminal and run this script again"
    echo -e "\n${CYAN}After installation, ensure nvcc is in your PATH.${NC}"
    
    # Try to open the download page
    if command -v xdg-open &> /dev/null; then
        xdg-open "https://developer.nvidia.com/cuda-downloads" 2>/dev/null || true
    elif command -v open &> /dev/null; then
        open "https://developer.nvidia.com/cuda-downloads" 2>/dev/null || true
    fi
    
    return 1
}

# ============================================================
# Vulkan Installation Check and Setup
# ============================================================
test_vulkan_installed() {
    if command -v vulkaninfo &> /dev/null; then
        VERSION=$(vulkaninfo --version 2>&1 | head -1)
        echo -e "${GREEN}Vulkan is installed: $VERSION${NC}"
        return 0
    fi
    
    # Check for Vulkan ICD loaders
    if [ -d "/usr/share/vulkan/icd.d" ] || [ -d "/etc/vulkan/icd.d" ]; then
        echo -e "${GREEN}Vulkan ICD files found.${NC}"
        return 0
    fi
    
    return 1
}

install_vulkan() {
    echo -e "\n${YELLOW}[Vulkan Setup]${NC}"
    
    if test_vulkan_installed; then
        echo -e "${GREEN}Vulkan runtime is already configured.${NC}"
        return 0
    fi
    
    if [ "$SKIP_INSTALL" = true ]; then
        echo -e "${YELLOW}Vulkan not found. Skipping installation.${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Vulkan runtime not found.${NC}"
    echo -e "\n${CYAN}To install Vulkan SDK:${NC}"
    echo -e "  1. Visit: https://vulkan.lunarg.com/sdk/home"
    echo -e "  2. Download the Windows SDK installer"
    echo -e "  3. Run the installer"
    echo -e "  4. Restart your terminal and run this script again"
    
    # Try to open the download page
    if command -v xdg-open &> /dev/null; then
        xdg-open "https://vulkan.lunarg.com/sdk/home" 2>/dev/null || true
    elif command -v open &> /dev/null; then
        open "https://vulkan.lunarg.com/sdk/home" 2>/dev/null || true
    fi
    
    return 1
}

# ============================================================
# GPU Backend Compilation Check
# ============================================================
check_gpu_backend_compilation() {
    local BACKEND=$1
    local FEATURES=""
    
    if [ "$BACKEND" = "cuda" ]; then
        FEATURES="--features cuda"
    elif [ "$BACKEND" = "vulkan" ]; then
        FEATURES="--features vulkan"
    fi
    
    echo -e "\n${YELLOW}[Checking $BACKEND Backend Compilation]${NC}"
    
    cd "$PROJECT_ROOT/core"
    if cargo check $FEATURES &> /dev/null; then
        echo -e "${GREEN}$BACKEND backend compiles successfully!${NC}"
        cd "$PROJECT_ROOT"
        return 0
    else
        echo -e "${YELLOW}$BACKEND backend compilation has issues (may need $BACKEND toolkit)${NC}"
        cd "$PROJECT_ROOT"
        return 1
    fi
}

# ============================================================
# Main Setup Logic
# ============================================================

# If CheckOnly, just check everything and exit
if [ "$CHECK_ONLY" = true ]; then
    echo -e "\n${CYAN}=== Environment Check ===${NC}"
    
    install_ffmpeg 2>/dev/null || true
    test_cuda_installed || true
    test_vulkan_installed || true
    
    echo -e "\n${CYAN}=== Summary ===${NC}"
    echo "Run this script with --help for more options"
    exit 0
fi

# Install all if requested
if [ "$INSTALL_ALL" = true ]; then
    INSTALL_FFMPEG=true
    INSTALL_CUDA=true
    INSTALL_VULKAN=true
fi

# Run installations
FFMPEG_INSTALLED=false
CUDA_INSTALLED=false
VULKAN_INSTALLED=false

if [ "$INSTALL_FFMPEG" = true ] || [ "$INSTALL_ALL" = true ]; then
    install_ffmpeg && FFMPEG_INSTALLED=true
fi

if [ "$INSTALL_CUDA" = true ] || [ "$INSTALL_ALL" = true ]; then
    install_cuda && CUDA_INSTALLED=true
fi

if [ "$INSTALL_VULKAN" = true ] || [ "$INSTALL_ALL" = true ]; then
    install_vulkan && VULKAN_INSTALLED=true
fi

# Check GPU backends
echo -e "\n${CYAN}=== Backend Compilation Status ===${NC}"

CUDA_AVAILABLE=false
VULKAN_AVAILABLE=false

if check_gpu_backend_compilation "cuda"; then
    CUDA_AVAILABLE=true
fi

if check_gpu_backend_compilation "vulkan"; then
    VULKAN_AVAILABLE=true
fi

# ============================================================
# Final Summary and Instructions
# ============================================================
echo -e "\n${CYAN}$(printf '=%.0s' {1..60})${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${CYAN}$(printf '=%.0s' {1..60})${NC}"

echo -e "\n${CYAN}[Installation Status]${NC}"
echo -e "FFmpeg: $(if [ "$FFMPEG_INSTALLED" = true ]; then echo -e "${GREEN}Installed${NC}"; else echo -e "${YELLOW}Not installed${NC}"; fi)"
echo -e "CUDA Toolkit: $(if [ "$CUDA_INSTALLED" = true ]; then echo -e "${GREEN}Installed${NC}"; else echo -e "${YELLOW}Not installed${NC}"; fi)"
echo -e "Vulkan SDK: $(if [ "$VULKAN_INSTALLED" = true ]; then echo -e "${GREEN}Installed${NC}"; else echo -e "${YELLOW}Not installed${NC}"; fi)"

echo -e "\n${CYAN}[Backend Compilation Status]${NC}"
echo -e "CUDA Backend: $(if [ "$CUDA_AVAILABLE" = true ]; then echo -e "${GREEN}Ready${NC}"; else echo -e "${YELLOW}Not ready${NC}"; fi)"
echo -e "Vulkan Backend: $(if [ "$VULKAN_AVAILABLE" = true ]; then echo -e "${GREEN}Ready${NC}"; else echo -e "${YELLOW}Not ready${NC}"; fi)"

echo -e "\n${CYAN}[Next Steps]${NC}"

if [ "$FFMPEG_INSTALLED" = false ]; then
    echo -e "${YELLOW}1. Install FFmpeg for video encoding support${NC}"
    echo -e "   Run: ./setup.sh --install-ffmpeg"
fi

if [ "$CUDA_INSTALLED" = false ] && [ "$VULKAN_INSTALLED" = false ]; then
    echo -e "${YELLOW}2. Install GPU backend for acceleration:${NC}"
    echo -e "   - CUDA: Visit https://developer.nvidia.com/cuda-downloads"
    echo -e "   - Vulkan: Visit https://vulkan.lunarg.com/sdk/home"
fi

if [ "$CUDA_AVAILABLE" = true ] || [ "$VULKAN_AVAILABLE" = true ]; then
    echo -e "\n${GREEN}To build with GPU backend:${NC}"
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo -e "   Release build: ./build.sh -r --with-cuda"
    fi
    if [ "$VULKAN_AVAILABLE" = true ]; then
        echo -e "   Vulkan build: ./build.sh -r --with-vulkan"
    fi
fi

echo -e "\n${CYAN}To use CPU backend (no GPU needed):${NC}"
echo -e "   Release build: ./build.sh -r"
echo -e "   Note: CPU inference is slower for large models"

echo -e "\n${CYAN}[Environment Variables]${NC}"
echo -e "Set VIDEO_MODEL_PATH to your GGUF model file:"
echo -e '   export VIDEO_MODEL_PATH="models/ltx-2.3-22b-dev-Q4_K_M.gguf"'

echo ""
