package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
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
		fmt.Println()
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: Model file not found: %s\n", modelPath)
		fmt.Fprintf(os.Stderr, "Set VIDEO_MODEL_PATH environment variable or use -m option\n")
		os.Exit(1)
	}

	// Find Rust binary
	rustBin := findRustBinary()
	if rustBin == "" {
		fmt.Fprintf(os.Stderr, "Error: Rust binary not found\n")
		fmt.Fprintf(os.Stderr, "Build with: cd core && cargo build --release\n")
		os.Exit(1)
	}

	fmt.Printf("Loading model: %s\n", modelPath)

	// Build arguments for Rust binary
	rustArgs := []string{
		"-m", modelPath,
		"-p", prompt,
		"-o", outputPath,
		"-f", strconv.Itoa(frames),
		"-W", strconv.Itoa(width),
		"-H", strconv.Itoa(height),
		"--fps", strconv.Itoa(fps),
		"--steps", strconv.Itoa(steps),
		"--sampler", sampler,
		"--cfg", cfgStr,
	}

	// Only add seed if it's not the default (empty string means use default/random)
	if seedStr != "" && seedStr != "-1" {
		rustArgs = append(rustArgs, "--seed", seedStr)
	}

	if negativePrompt != "" {
		rustArgs = append(rustArgs, "-n", negativePrompt)
	}

	if verbose {
		rustArgs = append(rustArgs, "-v")
	}

	cmd := exec.Command(rustBin, rustArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin

	err := cmd.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: Generation failed: %v\n", err)
		os.Exit(1)
	}
}

func findRustBinary() string {
	// Check common locations
	paths := []string{
		"core/target/release/video.exe",
		"core/target/debug/video.exe",
		"../core/target/release/video.exe",
		"../core/target/debug/video.exe",
	}

	// Check if we're in the project root
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exe)
		paths = append(paths, filepath.Join(dir, "core/target/release/video.exe"))
		paths = append(paths, filepath.Join(dir, "core/target/debug/video.exe"))
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// Try to find via cargo
	cmd := exec.Command("cargo", "locate-project", "--manifest-path", "core/Cargo.toml")
	output, err := cmd.Output()
	if err == nil {
		// Parse output to get project directory
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "Cargo.toml") {
				dir := filepath.Dir(line)
				testPath := filepath.Join(dir, "target/release/video.exe")
				if _, err := os.Stat(testPath); err == nil {
					return testPath
				}
			}
		}
	}

	return ""
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
  -W, --width <n>          Width (default: 512)
  -H, --height <n>         Height (default: 512)
  --fps <n>                FPS (default: 24)
  --steps <n>             Diffusion steps (default: 30)
  --sampler <name>         Sampler: euler, ddim, dpm++, rectified_flow (default: euler)
  --cfg <f>               CFG scale (default: 7.5)
  --seed <n>              Random seed (-1 for random)
  --backend <name>        Backend: auto, cpu, cuda, vulkan (default: auto)
  -v, --verbose            Verbose output
  -h, --help              Show this help

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
`)
}
