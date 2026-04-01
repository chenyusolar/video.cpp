package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	modelPath := getEnvOr("VIDEO_MODEL_PATH", "")
	prompt := getArg("-p", "--prompt", "")
	negativePrompt := getArg("-n", "--negative", "")
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

	if !hasArg("-m", "--model") && modelPath == "" {
		printUsage()
		os.Exit(0)
	}

	modelPath = getArgOr("-m", "--model", modelPath)

	if prompt == "" {
		fmt.Fprintf(os.Stderr, "Error: --prompt is required\n")
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
		fmt.Printf("Backend:   %s\n", backend)
		fmt.Println()
	}

	cfg, _ := strconv.ParseFloat(cfgStr, 32)
	seed, _ := strconv.ParseInt(seedStr, 10, 64)

	fmt.Printf("Loading model: %s\n", modelPath)
	fmt.Printf("(Native Go SDK requires compiled Rust core library)\n")
	fmt.Printf("\nConfiguration:\n")
	fmt.Printf("  Prompt: %s\n", prompt)
	fmt.Printf("  Frames: %d, Resolution: %dx%d\n", frames, width, height)
	fmt.Printf("  Steps: %d, Sampler: %s, CFG: %.1f\n", steps, sampler, cfg)
	fmt.Printf("  Seed: %d\n", seed)
	fmt.Printf("\nTo use the native Go SDK, build the Rust core first:\n")
	fmt.Printf("  cd core && cargo build --release\n")
	fmt.Printf("  cd cli && go build -o ../bin/video .\n")
}

func getEnvOr(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getArg(prefix, longPrefix, defaultVal string) string {
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
	if val := getArg(prefix, longPrefix, ""); val != "" {
		return val
	}
	return defaultVal
}

func getIntArgOr(prefix, longPrefix string, defaultVal int) int {
	val := getArg(prefix, longPrefix, "")
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
  -m, --model <path>       Path to model file (.gguv) [required]
  -p, --prompt <text>     Text prompt for generation [required]
  -n, --negative <text>    Negative prompt (optional)
  -o, --output <path>      Output video path (default: output.mp4)
  -f, --frames <n>         Number of frames (default: 24)
  -W, --width <n>          Width (default: 512)
  -H, --height <n>         Height (default: 512)
  --fps <n>                FPS (default: 24)
  --steps <n>              Diffusion steps (default: 30)
  --sampler <name>        Sampler: euler, ddim, dpm++ (default: euler)
  --cfg <f>                CFG scale (default: 7.5)
  --seed <n>               Random seed (-1 for random)
  --backend <name>         Backend: auto, cpu, cuda, vulkan (default: auto)
  -v, --verbose            Verbose output
  -h, --help               Show this help

Environment Variables:
  VIDEO_MODEL_PATH         Default model path
  VIDEO_BACKEND            Default backend (auto, cpu, cuda, vulkan)
  VIDEO_STEPS              Default diffusion steps
  VIDEO_SAMPLER            Default sampler

Examples:
  video -m ltx2.gguv -p "a dragon flying over city" -f 48 -o dragon.mp4
  video -m ltx2.gguv -p "cyberpunk street" --steps 50 --cfg 8.0 -v

Build from source:
  cd core && cargo build --release
  cd cli && go build -o ../bin/video .
`)
}
