
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "print.h"
#include "bunnyIO.h"
#include "device-bunnyMIP.h"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                         __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)


__global__ void threshold_kernel(const uint16_t* input, uint16_t* output,
                                 uint16_t threshold, int total_voxels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        uint16_t v = input[idx];
        output[idx] = (v < threshold) ? 0 : v;
    }
}


__global__ void gaussian_blur_kernel(const uint16_t* input, uint16_t* output,
                                     const float* kernel,
                                     int N, int M)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= N || y >= N || z >= M) return;

    float sum = 0.0f;
    int ki = 0;

    for (int dz = -1; dz <= 1; ++dz) {
        int nz = max(0, min(M - 1, z + dz));
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = max(0, min(N - 1, y + dy));
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = max(0, min(N - 1, x + dx));
                uint16_t v = input[(size_t)nz * N * N + ny * N + nx];
                sum += (float)v * kernel[ki++];
            }
        }
    }

    output[(size_t)z * N * N + y * N + x] = (uint16_t)sum;
}


__global__ void mip_kernel(const uint16_t* volume, uint16_t* images,
                           const float* R, int N, int M, int ray_range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N) return;

    float u = x - N / 2.0f;
    float v = y - N / 2.0f;
    uint16_t max_val = 0;

    for (int step = -ray_range; step < ray_range; ++step) {
        float w = (float)step;

        float rotX = R[0] * u + R[1] * v + R[2] * w + N / 2.0f;
        float rotY = R[3] * u + R[4] * v + R[5] * w + N / 2.0f;
        float rotZ = R[6] * u + R[7] * v + R[8] * w + M / 2.0f;

        if (rotX >= 0 && rotX < N &&
            rotY >= 0 && rotY < N &&
            rotZ >= 0 && rotZ < M) {
            size_t idx = (size_t)((int)rotZ * N * N + (int)rotY * N + (int)rotX);
            uint16_t val = volume[idx];
            if (val > max_val) max_val = val;
        }
    }

    images[y * N + x] = max_val;
}

__global__ void multiple_mip_kernel(const uint16_t* volume, uint16_t* images, uint16_t num_images,
                                    const float* R, int N, int M, int ray_range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img_index = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= N || y >= N || img_index >= num_images) return;

    float u = x - N / 2.0f;
    float v = y - N / 2.0f;
    
    uint16_t max_val = 0;
    int r_i = img_index * 9;
    
    for (int step = -ray_range; step < ray_range; ++step) {
        float w = (float)step;

        float rotX = R[r_i + 0] * u + R[r_i + 1] * v + R[r_i + 2] * w + N / 2.0f;
        float rotY = R[r_i + 3] * u + R[r_i + 4] * v + R[r_i + 5] * w + N / 2.0f;
        float rotZ = R[r_i + 6] * u + R[r_i + 7] * v + R[r_i + 8] * w + M / 2.0f;

        if (rotX >= 0 && rotX < N &&
            rotY >= 0 && rotY < N &&
            rotZ >= 0 && rotZ < M) {
            size_t idx = (size_t)((int)rotZ * N * N + (int)rotY * N + (int)rotX);
            uint16_t val = volume[idx];
            if (val > max_val) max_val = val;
        }
    }

    images[img_index * N * N + y * N + x] = max_val;
}


// =============================================================================
// Device entry point.
// =============================================================================
void device_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output)
{
    print("Running functions on the GPU\n");

    const int N = kBunnySize;
    const int M = kBunnyN;
    size_t vol_bytes = (size_t)N * N * M * sizeof(uint16_t);
    size_t out_bytes = (size_t)N * N * sizeof(uint16_t);
    int total_voxels = N * N * M;

    print("  gpu: allocating %.0f MB of VRAM\n",
          (3.0 * vol_bytes + out_bytes) / (1024.0 * 1024.0));

    // --- Device allocations ---
    uint16_t *d_input, *d_thresholded, *d_blurred, *d_output;
    float *d_R, *d_kernel;

    CUDA_CHECK(cudaMalloc(&d_input,       vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_thresholded, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_blurred,     vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,      out_bytes));
    CUDA_CHECK(cudaMalloc(&d_R,           9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel,      27 * sizeof(float)));

    // --- Host → Device transfers ---
    CUDA_CHECK(cudaMemcpy(d_input, input, vol_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R,    R,     9 * sizeof(float), cudaMemcpyHostToDevice));

    // --- Precompute Gaussian kernel (3×3×3) ---
    {
        float h_kernel[27];
        float kernel_sum = 0.0f;
        int ki = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float d2 = (float)(dx*dx + dy*dy + dz*dz);
                    float w = expf(-d2 / (2.0f * sigma * sigma));
                    h_kernel[ki++] = w;
                    kernel_sum += w;
                }
            }
        }
        for (int i = 0; i < 27; ++i) h_kernel[i] /= kernel_sum;
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, 27 * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    dim3 blur_block(8, 8, 4);
    dim3 blur_grid((N + 7) / 8, (N + 7) / 8, (M + 3) / 4);

    dim3 mip_block(16, 16);
    dim3 mip_grid((N + 15) / 16, (N + 15) / 16);

    int thresh_block = 256;
    int thresh_grid = (total_voxels + thresh_block - 1) / thresh_block;

    // --- Pipeline ---

    // Step 1: Threshold
    print("  gpu: applying threshold\n");
    threshold_kernel<<<thresh_grid, thresh_block>>>(d_input, d_thresholded,
                                                     threshold, total_voxels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Gaussian Blur
    print("  gpu: applying filter\n");
    gaussian_blur_kernel<<<blur_grid, blur_block>>>(d_thresholded, d_blurred,
                                                     d_kernel, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: MIP Projection
    print("  gpu: generating MIP\n");
    int ray_range = (int)(sqrtf((float)(N*N + N*N + M*M)) / 2.0f) + 1;
    mip_kernel<<<mip_grid, mip_block>>>(d_blurred, d_output, d_R, N, M, ray_range);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Device → Host transfer ---
    CUDA_CHECK(cudaMemcpy(output, d_output, out_bytes, cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_thresholded));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_kernel));
}


// =============================================================================
// Device entry point - batch rendering.
// =============================================================================
void device_bunny_multiple_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output, uint16_t num_images)
{
    print("Running functions on the GPU - multiple images at the same time\n");

    const int N = kBunnySize;
    const int M = kBunnyN;
    size_t vol_bytes = (size_t)N * N * M * sizeof(uint16_t);
    size_t out_bytes = (size_t)N * N * sizeof(uint16_t) * num_images;
    int total_voxels = N * N * M;

    print("  gpu: allocating %.0f MB of VRAM\n",
          (3.0 * vol_bytes + out_bytes) / (1024.0 * 1024.0));

    // --- Device allocations ---
    uint16_t *d_input, *d_thresholded, *d_blurred, *d_output;
    float *d_R, *d_kernel;

    CUDA_CHECK(cudaMalloc(&d_input,       vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_thresholded, vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_blurred,     vol_bytes));
    CUDA_CHECK(cudaMalloc(&d_output,      out_bytes));
    CUDA_CHECK(cudaMalloc(&d_R,           9 * sizeof(float) * num_images));
    CUDA_CHECK(cudaMalloc(&d_kernel,      27 * sizeof(float)));

    // --- Host → Device transfers ---
    CUDA_CHECK(cudaMemcpy(d_input, input, vol_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R,    R,     9 * sizeof(float) * num_images, cudaMemcpyHostToDevice));

    // --- Precompute Gaussian kernel (3×3×3) ---
    {
        float h_kernel[27];
        float kernel_sum = 0.0f;
        int ki = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float d2 = (float)(dx*dx + dy*dy + dz*dz);
                    float w = expf(-d2 / (2.0f * sigma * sigma));
                    h_kernel[ki++] = w;
                    kernel_sum += w;
                }
            }
        }
        for (int i = 0; i < 27; ++i) h_kernel[i] /= kernel_sum;
        CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, 27 * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    dim3 blur_block(8, 8, 4);
    dim3 blur_grid((N + 7) / 8, (N + 7) / 8, (M + 3) / 4);

    dim3 mip_block(8, 8, 4);
    dim3 mip_grid((N + 7) / 8, (N + 7) / 8, (num_images + 3) / 4);

    int thresh_block = 256;
    int thresh_grid = (total_voxels + thresh_block - 1) / thresh_block;

    // --- Pipeline ---

    // Step 1: Threshold
    print("  gpu: applying threshold\n");
    threshold_kernel<<<thresh_grid, thresh_block>>>(d_input, d_thresholded,
                                                     threshold, total_voxels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Gaussian Blur
    print("  gpu: applying filter\n");
    gaussian_blur_kernel<<<blur_grid, blur_block>>>(d_thresholded, d_blurred,
                                                     d_kernel, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: MIP Projection
    print("  gpu: generating MIP\n");
    int ray_range = (int)(sqrtf((float)(N*N + N*N + M*M)) / 2.0f) + 1;
    multiple_mip_kernel<<<mip_grid, mip_block>>>(d_blurred, d_output, num_images, d_R, N, M, ray_range);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Device → Host transfer ---
    CUDA_CHECK(cudaMemcpy(output, d_output, out_bytes, cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_thresholded));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_kernel));
}