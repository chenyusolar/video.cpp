#!/bin/bash
set -e

echo "========================================="
echo " video.cpp Build Script"
echo "========================================="

cd "$(dirname "$0")/.."

if [ ! -d "core" ]; then
    echo "Error: core directory not found"
    exit 1
fi

echo "[1/4] Building Rust core..."
cd core

if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found in core/"
    exit 1
fi

cargo build --release 2>&1 | tail -20

if [ $? -ne 0 ]; then
    echo "Rust build failed"
    exit 1
fi

echo "[2/4] Rust core built successfully"
echo ""

echo "[3/4] Creating shared library..."
cd target/release

if [ -f "libvideo_core.a" ]; then
    echo "Static library already exists"
elif [ -f "libvideo_core.so" ]; then
    echo "Shared library already exists"
elif [ -f "libvideo_core.dylib" ]; then
    echo "Shared library already exists"
else
    echo "Warning: No library found in target/release/"
    ls -la
fi

cd ../..

echo "[4/4] Build complete!"
echo ""

echo "To build the CLI:"
echo "  cd cli && go build -o ../bin/video ."
echo ""
echo "To run an example:"
echo "  cd go/examples/simple && go run main.go"
echo ""

echo "Available targets:"
echo "  bin/video          - CLI tool"
echo "  core/target/release/libvideo_core.* - Rust core library"
