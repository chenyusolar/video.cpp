#ifndef VIDEO_H
#define VIDEO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

typedef uint64_t model_handle;
typedef uint64_t tensor_handle;

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
} generate_request;

typedef struct {
    uint8_t* data;
    size_t size;
    int32_t width;
    int32_t height;
    int32_t fps;
    int64_t generation_time_ms;
} video_output;

typedef enum {
    VIDEO_BACKEND_CPU = 0,
    VIDEO_BACKEND_CUDA = 1,
    VIDEO_BACKEND_VULKAN = 2,
} video_backend;

typedef enum {
    VIDEO_OK = 0,
    VIDEO_ERROR_LOAD_FAILED = 1,
    VIDEO_ERROR_INVALID_PARAM = 2,
    VIDEO_ERROR_GENERATION_FAILED = 3,
    VIDEO_ERROR_OUT_OF_MEMORY = 4,
    VIDEO_ERROR_BACKEND_ERROR = 5,
    VIDEO_ERROR_UNSUPPORTED = 6,
} video_error;

video_error video_load(const char* model_path, model_handle* out_handle);

void video_free(model_handle handle);

video_error video_generate(model_handle handle, generate_request req, video_output* out);

video_error video_generate_image_to_video(
    model_handle handle,
    const uint8_t* init_image,
    size_t image_size,
    const char* prompt,
    int32_t width,
    int32_t height,
    int32_t frames,
    float strength,
    int32_t steps,
    float cfg_scale,
    int64_t seed,
    video_output* out
);

video_error video_generate_video_to_video(
    model_handle handle,
    const uint8_t* init_video,
    size_t video_size,
    const char* prompt,
    int32_t width,
    int32_t height,
    int32_t frames,
    float strength,
    int32_t steps,
    float cfg_scale,
    int64_t seed,
    video_output* out
);

video_error video_set_backend(video_backend backend);

const char* video_get_version(void);

video_error video_get_memory_info(int64_t* allocated_bytes, int64_t* reserved_bytes);

#ifdef __cplusplus
}
#endif

#endif // VIDEO_H
