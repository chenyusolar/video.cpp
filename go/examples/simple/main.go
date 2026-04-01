package main

import (
	"fmt"
	"log"
	"os"

	video "github.com/video-ai/video.cpp/go"
)

func main() {
	log.Println("video.cpp Example - Simple Video Generation")

	modelPath := os.Getenv("VIDEO_MODEL_PATH")
	if modelPath == "" {
		modelPath = "models/ltx2.gguv"
	}

	client, err := video.Load(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer client.Close()

	fmt.Printf("Model loaded successfully\n")
	fmt.Printf("Version: %s\n", video.GetVersion())

	req := video.GenerateRequest{
		Prompt:   "a beautiful sunset over the ocean",
		Frames:   24,
		Width:    512,
		Height:   512,
		FPS:      24,
		Steps:    30,
		Sampler:  "euler",
		CFGScale: 7.5,
		Seed:     42,
	}

	fmt.Println("Generating video...")
	fmt.Printf("Prompt: %s\n", req.Prompt)
	fmt.Printf("Frames: %d, Resolution: %dx%d\n", req.Frames, req.Width, req.Height)
	fmt.Printf("Steps: %d, Sampler: %s\n", req.Steps, req.Sampler)

	result, err := client.Generate(req)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	fmt.Printf("Generation completed in %dms\n", result.GenerationTimeMs)
	fmt.Printf("Output: %dx%d @ %dfps, %d bytes\n", result.Width, result.Height, result.FPS, len(result.Data))

	if err := os.WriteFile("output.mp4", result.Data, 0644); err != nil {
		log.Fatalf("Failed to write video file: %v", err)
	}
	fmt.Println("Video saved to output.mp4")
}
