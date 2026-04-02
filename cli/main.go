package main

/*
#cgo LDFLAGS:-L${SRCDIR}/../core/target/release -lvideo_core
#include <stdlib.h>
#include "../include/video.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    int32_t frames;
    int32_t width;
    int32_t height;
    int32_t fps;
    int32_t steps;
    const char* sampler;
    float cfg_scale;
    int64_t seed;
    int32_t device_id;
} GoGenerateRequest;

typedef struct {
    uint8_t* data;
    size_t size;
    int32_t width;
    int32_t height;
    int32_t fps;
    int64_t generation_time_ms;
} GoVideoOutput;
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"unsafe"
)

func main() {
	modelPath := getEnvOr("VIDEO_MODEL_PATH", "")
	prompt := getArg("-p", "--prompt")
	negativePrompt := getArg("-n", "--negative")
	outputPath := getArgOr("-o", "--output", "output.mp4")
	frames := getIntArgOr("-f", "--frames", 24)
	width := getIntArgOr("-W", "--width", 512)
	height := getIntArgOr("-H", "--height", 512)
	fps := getIntArgOr("--fps", "", 24)
	steps := getIntArgOr("--steps", "", 30)
	sampler := getArgOr("--sampler", "", "euler")
	cfgStr := getArgOr("--cfg", "", "7.5")
	seedStr := getArgOr("--seed", "", "-1")
	backend := getArgOr("--backend", "", "auto")
	verbose := hasArg("-v", "--verbose")

	if hasArg("-h", "--help") {
		printUsage()
		os.Exit(0)
	}

	if !hasArg("-m", "--model") && modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: --model is required\n")
		fmt.Fprintf(os.Stderr, "Use --help for usage information\n")
		os.Exit(1)
	}

	modelPath = getArgOr("-m", "--model", modelPath)

	if prompt == "" {
		fmt.Fprintf(os.Stderr, "Error: --prompt is required\n")
		fmt.Fprintf(os.Stderr, "Use --help for usage information\n")
		os.Exit(1)
	}

	cfg, _ := strconv.ParseFloat(cfgStr, 32)
	seed, _ := strconv.ParseInt(seedStr, 10, 64)

	if verbose {
		fmt.Printf("video.cpp CLI - Local Video Generation Engine\n")
		fmt.Printf("=============================================\n")
		fmt.Printf("Model:     %s\n", modelPath)
		fmt.Printf("Prompt:    %s\n", prompt)
		if negativePrompt != "" {
			fmt.Printf("Negative:  %s\n", negativePrompt)
		}
		fmt.Printf("Output:    %s\n", outputPath)
		fmt.Printf("Frames:    %d\n", frames)
		fmt.Printf("Resolution:%dx%d\n", width, height)
		fmt.Printf("FPS:       %d\n", fps)
		fmt.Printf("Steps:     %d\n", steps)
		fmt.Printf("Sampler:   %s\n", sampler)
		fmt.Printf("CFG:       %s\n", cfgStr)
		fmt.Printf("Seed:      %s\n", seedStr)
		fmt.Printf("Backend:   %s\n", backend)
		fmt.Println()
	}

	// Check if model file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: Model file not found: %s\n", modelPath)
		fmt.Fprintf(os.Stderr, "Set VIDEO_MODEL_PATH environment variable or use -m option\n")
		os.Exit(1)
	}

	// Get absolute path for model
	absModelPath, err := filepath.Abs(modelPath)
	if err != nil {
		absModelPath = modelPath
	}

	// Set backend
	if backend != "auto" {
		os.Setenv("VIDEO_BACKEND", backend)
	}

	fmt.Printf("Loading model: %s\n", absModelPath)

	// Load the model
	cModelPath := C.CString(absModelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	var handle C.model_handle
	result := C.video_load(cModelPath, &handle)

	if result != 0 {
		fmt.Fprintf(os.Stderr, "Error: Failed to load model (error code: %d)\n", result)
		fmt.Fprintf(os.Stderr, "Make sure the Rust core library is built:\n")
		fmt.Fprintf(os.Stderr, "  cd core && cargo build --release\n")
		os.Exit(1)
	}

	defer C.video_free(handle)

	fmt.Printf("Model loaded successfully!\n")
	fmt.Printf("Starting video generation...\n\n")

	// Create generate request
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	var cNegPrompt *C.char
	if negativePrompt != "" {
		cNegPrompt = C.CString(negativePrompt)
		defer C.free(unsafe.Pointer(cNegPrompt))
	}

	cSampler := C.CString(sampler)
	defer C.free(unsafe.Pointer(cSampler))

	req := C.GoGenerateRequest{
		prompt:          cPrompt,
		negative_prompt: cNegPrompt,
		frames:          C.int(frames),
		width:           C.int(width),
		height:          C.int(height),
		fps:             C.int(fps),
		steps:           C.int(steps),
		sampler:         cSampler,
		cfg_scale:       C.float(float32(cfg)),
		seed:            C.long(seed),
		device_id:       C.int(0),
	}

	var output C.GoVideoOutput

	result = C.video_generate(handle, req, &output)

	if result != 0 {
		fmt.Fprintf(os.Stderr, "Error: Generation failed (error code: %d)\n", result)
		os.Exit(1)
	}

	// Copy output data to Go
	videoData := C.GoBytes(unsafe.Pointer(output.data), C.size_t(output.size))
	C.free(unsafe.Pointer(output.data))

	fmt.Printf("\nGeneration complete!\n")
	fmt.Printf("  Resolution: %dx%d\n", output.width, output.height)
	fmt.Printf("  FPS: %d\n", output.fps)
	fmt.Printf("  Frames: %d bytes\n", output.size)
	fmt.Printf("  Time: %dms\n", output.generation_time_ms)

	// Write output file
	fmt.Printf("\nWriting video to: %s\n", outputPath)

	// Determine format from extension
	ext := strings.ToLower(filepath.Ext(outputPath))
	switch ext {
	case ".mp4":
		// For now, write raw bytes - in future, encode with ffmpeg
		if err := os.WriteFile(outputPath, videoData, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
			os.Exit(1)
		}
	case ".rgb", ".raw":
		if err := os.WriteFile(outputPath, videoData, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
			os.Exit(1)
		}
	default:
		// Try to detect format or default to raw
		if err := os.WriteFile(outputPath, videoData, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Video saved successfully!\n")
}

func getEnvOr(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getArg(prefix, longPrefix string) string {
	for i, arg := range os.Args {
		if arg == prefix || arg == longPrefix {
			if i+1 < len(os.Args) {
				return os.Args[i+1]
			}
		}
	}
	return ""
}

func getArgOr(prefix, longPrefix, defaultVal string) string {
	if val := getArg(prefix, longPrefix); val != "" {
		return val
	}
	return defaultVal
}

func getIntArgOr(prefix, longPrefix string, defaultVal int) int {
	val := getArg(prefix, longPrefix)
	if val == "" {
		return defaultVal
	}
	if i, err := strconv.Atoi(val); err == nil {
		return i
	}
	return defaultVal
}

func hasArg(prefix, longPrefix string) bool {
	for _, arg := range os.Args {
		if arg == prefix || arg == longPrefix {
			return true
		}
	}
	return false
}

func printUsage() {
	fmt.Println(`video.cpp - Local Video Generation Engine

Usage: video [options]

Options:
  -m, --model <path>       Path to model file (.gguf) [required]
  -p, --prompt <text>      Text prompt for generation [required]
  -n, --negative <text>   Negative prompt (optional)
  -o, --output <path>      Output video path (default: output.mp4)
  -f, --frames <n>         Number of frames (default: 24)
  -W, --width <n>         Width (default: 512)
  -H, --height <n>        Height (default: 512)
  --fps <n>               FPS (default: 24)
  --steps <n>             Diffusion steps (default: 30)
  --sampler <name>        Sampler: euler, ddim, dpm++, rectified_flow (default: euler)
  --cfg <f>               CFG scale (default: 7.5)
  --seed <n>              Random seed (-1 for random)
  --backend <name>         Backend: auto, cpu, cuda, vulkan (default: auto)
  -v, --verbose            Verbose output
  -h, --help               Show this help

Environment Variables:
  VIDEO_MODEL_PATH         Default model path
  VIDEO_BACKEND            Default backend (auto, cpu, cuda, vulkan)
  VIDEO_STEPS              Default diffusion steps
  VIDEO_SAMPLER            Default sampler

Examples:
  video -m ltx2.gguf -p "a dragon flying over city" -f 48 -o dragon.mp4
  video -m ltx2.gguf -p "cyberpunk street" --steps 50 --cfg 8.0 -v

Build from source:
  cd core && cargo build --release
  cd cli && go build -o ../bin/video .
`)
}
