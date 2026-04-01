package internal

/*
#cgo LDFLAGS:-L${SRCDIR}/../../core/target/release -lvideo_core
#include <stdlib.h>
#include "../../include/video.h"
#include <stdint.h>

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
	"errors"
	"unsafe"
)

type ModelHandle unsafe.Pointer
type VideoOutputInternal struct {
	Data             []byte
	Width            int
	Height           int
	FPS              int
	GenerationTimeMs int64
}

func LoadModel(path string) (ModelHandle, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var handle C.model_handle
	err := C.video_load(cPath, &handle)
	if err != 0 {
		return nil, errors.New("failed to load model")
	}
	return ModelHandle(handle), nil
}

func FreeModel(handle ModelHandle) error {
	C.video_free(C.model_handle(handle))
	return nil
}

type GenerateOptions struct {
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
	DeviceID       int
}

func GenerateVideo(handle ModelHandle, opts GenerateOptions) (*VideoOutputInternal, error) {
	cReq := C.GoGenerateRequest{
		prompt:          C.CString(opts.Prompt),
		negative_prompt: C.CString(opts.NegativePrompt),
		frames:          C.int32(opts.Frames),
		width:           C.int32(opts.Width),
		height:          C.int32(opts.Height),
		fps:             C.int32(opts.FPS),
		steps:           C.int32(opts.Steps),
		sampler:         C.CString(opts.Sampler),
		cfg_scale:       C.float(opts.CFGScale),
		seed:            C.int64_t(opts.Seed),
		device_id:       C.int32(opts.DeviceID),
	}
	defer C.free(unsafe.Pointer(cReq.prompt))
	defer C.free(unsafe.Pointer(cReq.negative_prompt))
	defer C.free(unsafe.Pointer(cReq.sampler))

	var cOutput C.GoVideoOutput
	err := C.video_generate(C.model_handle(handle), cReq, &cOutput)
	if err != 0 {
		return nil, errors.New("generation failed")
	}

	data := C.GoBytes(unsafe.Pointer(cOutput.data), C.size_t(cOutput.size))
	C.free(unsafe.Pointer(cOutput.data))

	return &VideoOutputInternal{
		Data:             data,
		Width:            int(cOutput.width),
		Height:           int(cOutput.height),
		FPS:              int(cOutput.fps),
		GenerationTimeMs: int64(cOutput.generation_time_ms),
	}, nil
}

type ModelInfoInternal struct {
	Name        string
	Type        string
	Parameters  uint64
	LatentShape []int
	HasVAE      bool
	HasAudio    bool
}

func GetModelInfo(handle ModelHandle) (*ModelInfoInternal, error) {
	return &ModelInfoInternal{
		Name:        "LTX-2",
		Type:        "ltx-video",
		Parameters:  14000000000,
		LatentShape: []int{1, 16, 64, 64},
		HasVAE:      true,
		HasAudio:    false,
	}, nil
}
