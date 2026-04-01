package video

import (
	"fmt"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/video-ai/video.cpp/go/internal"
)

type GenerateRequest struct {
	Prompt         string
	NegativePrompt string
	Frames         int
	Width          int
	Height         int
	FPS            int
	Steps          int
	Sampler        string
	CFGScale       float32
	Seed           int64
	Backend        string
}

type VideoOutput struct {
	Data             []byte
	Width            int
	Height           int
	FPS              int
	GenerationTimeMs int64
}

type Client struct {
	modelPath string
	handle    unsafe.Pointer
}

func Load(modelPath string) (*Client, error) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", modelPath)
	}

	handle, err := internal.LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return &Client{
		modelPath: modelPath,
		handle:    handle,
	}, nil
}

func (c *Client) Close() error {
	if c.handle != nil {
		return internal.FreeModel(c.handle)
	}
	return nil
}

func (c *Client) Generate(req GenerateRequest) (*VideoOutput, error) {
	return c.generateWithCallback(req, nil)
}

func (c *Client) GenerateWithCallback(req GenerateRequest, callback func(progress, total int)) (*VideoOutput, error) {
	if req.Width == 0 {
		req.Width = 512
	}
	if req.Height == 0 {
		req.Height = 512
	}
	if req.Frames == 0 {
		req.Frames = 24
	}
	if req.Steps == 0 {
		req.Steps = 30
	}
	if req.Sampler == "" {
		req.Sampler = "euler"
	}
	if req.CFGScale == 0 {
		req.CFGScale = 7.5
	}

	return c.generateWithCallback(req, callback)
}

func (c *Client) generateWithCallback(req GenerateRequest, callback func(progress, total int)) (*VideoOutput, error) {
	output, err := internal.GenerateVideo(c.handle, req)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	return &VideoOutput{
		Data:             output.Data,
		Width:            output.Width,
		Height:           output.Height,
		FPS:              output.FPS,
		GenerationTimeMs: output.GenerationTimeMs,
	}, nil
}

func (c *Client) GetInfo() (*ModelInfo, error) {
	info, err := internal.GetModelInfo(c.handle)
	if err != nil {
		return nil, err
	}
	return &ModelInfo{
		Name:        info.Name,
		Type:        info.Type,
		Parameters:  info.Parameters,
		LatentShape: info.LatentShape,
		HasVAE:      info.HasVAE,
		HasAudio:    info.HasAudio,
	}, nil
}

type ModelInfo struct {
	Name        string
	Type        string
	Parameters  uint64
	LatentShape []int
	HasVAE      bool
	HasAudio    bool
}

func (c *Client) SaveVideo(output *VideoOutput, path string) error {
	ext := filepath.Ext(path)
	switch ext {
	case ".mp4", ".MP4":
		return saveMP4(output, path)
	case ".webm", ".WEBM":
		return saveWebM(output, path)
	case ".gif", ".GIF":
		return saveGIF(output, path)
	default:
		return fmt.Errorf("unsupported format: %s (supported: mp4, webm, gif)", ext)
	}
}

func (v *VideoOutput) Save(path string) error {
	ext := filepath.Ext(path)
	switch ext {
	case ".mp4", ".MP4":
		return saveMP4(v, path)
	case ".webm", ".WEBM":
		return saveWebM(v, path)
	case ".gif", ".GIF":
		return saveGIF(v, path)
	default:
		return fmt.Errorf("unsupported format: %s (supported: mp4, webm, gif)", ext)
	}
}

func saveMP4(v *VideoOutput, path string) error {
	return fmt.Errorf("MP4 encoding not yet implemented - install ffmpeg for video encoding")
}

func saveWebM(v *VideoOutput, path string) error {
	return fmt.Errorf("WebM encoding not yet implemented")
}

func saveGIF(v *VideoOutput, path string) error {
	return fmt.Errorf("GIF encoding not yet implemented")
}
