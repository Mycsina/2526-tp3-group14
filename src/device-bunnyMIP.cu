
#include <cstdio>
#include <cstdlib>

#include "print.h"
#include "bunnyIO.h"
#include "device-bunnyMIP.h"


// TODO: Define your kernels here


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                         __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)


// =============================================================================
// Device entry point.
// =============================================================================
void device_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output)
{
    print("Running functions on the GPU\n");

    size_t vol_bytes = (size_t)kBunnySize * kBunnySize * kBunnyN * sizeof(uint16_t);
    size_t out_bytes = (size_t)kBunnySize * kBunnySize * sizeof(uint16_t);

    print("  gpu: allocating %.0f MB of VRAM\n", (3.0 * vol_bytes + out_bytes) / (1024.0 * 1024.0));

    // --- Device allocations ---
    uint16_t *d_input, *d_thresholded, *d_blurred, *d_output;
    float *d_R;

    CUDA_CHECK(cudaMalloc(&d_input,       vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_thresholded, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_blurred,     vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,      out_bytes));
    CUDA_CHECK(cudaMalloc(&d_R,           9 * sizeof(float)));

    // --- Host → Device transfers ---
    CUDA_CHECK(cudaMemcpy(d_input, input,   vol_bytes,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R,    R,       9 * sizeof(float), cudaMemcpyHostToDevice));

    // --- Pipeline ---

    // Step 1: Threshold
    print("  gpu: applying threshold\n");


    // Step 2: Gaussian Blur
    print("  gpu: applying filter\n");

    // Step 3: MIP Projection
    print("  gpu: generating MIP\n");

    // --- Device → Host transfer ---
    CUDA_CHECK(cudaMemcpy(output, d_output, out_bytes, cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_thresholded));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_R));
}
