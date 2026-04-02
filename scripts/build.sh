#!/bin/bash
# video.cpp Build Script
# Usage: ./build.sh [-r] [--with-cuda] [--with-vulkan] [--clean]

set -e

RELEASE=false
WITH_CUDA=false
WITH_VULKAN=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--release)
            RELEASE=true
            shift
            ;;
        --with-cuda)
            WITH_CUDA=true
            shift
            ;;
        --with-vulkan)
            WITH_VULKAN=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo " video.cpp Build Script"
echo "========================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "[Clean] Removing build artifacts..."
    rm -rf core/target
    rm -rf bin
    echo "[Clean] Done"
    echo ""
fi

# Build Rust core
echo "[1/3] Building Rust core..."

if [ ! -d "core" ]; then
    echo "Error: core directory not found"
    exit 1
fi

if [ ! -f "core/Cargo.toml" ]; then
    echo "Error: Cargo.toml not found in core/"
    exit 1
fi

FEATURES=""
BUILD_TYPE=""

if [ "$RELEASE" = true ]; then
    BUILD_TYPE="--release"
fi

if [ "$WITH_CUDA" = true ] && [ "$WITH_VULKAN" = true ]; then
    FEATURES="--features cuda,vulkan"
    echo "[Backend] Building with CUDA + Vulkan support..."
elif [ "$WITH_CUDA" = true ]; then
    FEATURES="--features cuda"
    echo "[Backend] Building with CUDA support..."
elif [ "$WITH_VULKAN" = true ]; then
    FEATURES="--features vulkan"
    echo "[Backend] Building with Vulkan support..."
else
    echo "[Backend] Building with CPU only (no GPU acceleration)"
fi

cd core
cargo build $BUILD_TYPE $FEATURES 2>&1 | tail -20

if [ $? -ne 0 ]; then
    echo "Rust build failed"
    exit 1
fi

echo "[Rust core] Built successfully"
echo ""

# Build CLI
echo "[2/3] Building Go CLI..."

if [ ! -d "$PROJECT_ROOT/cli" ]; then
    echo "Warning: CLI directory not found, skipping..."
else
    mkdir -p "$PROJECT_ROOT/bin"
    
    cd "$PROJECT_ROOT/cli"
    
    if [ "$RELEASE" = true ]; then
        go build -ldflags="-s -w" -o "$PROJECT_ROOT/bin/video" .
    else
        go build -o "$PROJECT_ROOT/bin/video" .
    fi
    
    if [ $? -ne 0 ]; then
        echo "Go build failed"
        exit 1
    fi
    
    echo "[CLI] Built successfully"
fi

echo ""
echo "[3/3] Build complete!"
echo "========================================="

echo ""
echo "[Outputs]"
echo "  CLI:   $PROJECT_ROOT/bin/video"
echo "  Core:  $PROJECT_ROOT/core/target/$(if [ "$RELEASE" = true ]; then echo "release"; else echo "debug"; fi)/libvideo_core.*"

echo ""
echo "[Backend Status]"
echo "  CPU:    Always available"
if [ "$WITH_CUDA" = true ]; then
    echo "  CUDA:   Enabled"
else
    echo "  CUDA:   Disabled (use --with-cuda to enable)"
fi
if [ "$WITH_VULKAN" = true ]; then
    echo "  Vulkan: Enabled"
else
    echo "  Vulkan: Disabled (use --with-vulkan to enable)"
fi

echo ""
echo "[Next Steps]"
echo "  Set model path:"
echo '    export VIDEO_MODEL_PATH="models/ltx-2.3-22b-dev-Q4_K_M.gguf"'
echo ""
echo "  Run:"
echo '    ./bin/video -p "a dragon flying"'
echo ""
echo "  For GPU acceleration:"
if [ "$WITH_CUDA" = true ]; then
    echo '    export VIDEO_BACKEND=cuda'
    echo '    ./bin/video -p "a dragon flying"'
fi
if [ "$WITH_VULKAN" = true ]; then
    echo '    export VIDEO_BACKEND=vulkan'
    echo '    ./bin/video -p "a dragon flying"'
fi
