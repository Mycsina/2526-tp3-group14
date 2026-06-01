# CLE - Practical Assignment 3

**Volumetric Maximum Intensity Projection (MIP) with CUDA Programming**

*Group 14*

[108269] André Cardoso

[112793] Cláudia Seabra

[135716] Viktor Bogojevic

## Instructions

Compiling the application: make

Running the application: `./bunnyMIP [--threshold ] [--sigma ] [--roll ] [--pitch ] [--yaw ] [--final_roll ] [--final_pitch ] [--final_yaw ]   [--steps ]`

- `--threshold`: Defines the intensity floor. Any voxel below this value is set to 0 (default: 32768).
- `--sigma`: Defines the standard deviation ($\sigma$) used to calculate the Gaussian weights for the blur filter (default: 1.0).
- `--roll`, `--pitch`, `--yaw`: Defines the (starting) rotation of the volume in degrees (default: 0.0 for all).
- `--final_roll`, `--final_pitch`, `--final_yaw`: Defines the final rotation of the volume in degrees (default: 0.0 for all).
- `--steps`: Defines how many in how many steps will the volume rotating from the starting rotation to the final rotation be rendered (default: 1)

**Validating the implementation:**
A validation script is provided to compare the GPU output against the CPU "gold standard" across multiple parameter sets.
`./validate.sh`

## Implementation strategy

The processing pipeline was adapted from the sequential CPU implementation to the GPU using the CUDA framework. The implementation is divided into distinct kernels mapping to the three main stages of the volume rendering pipeline, managed by the host function `device_bunny_mip()`.

**1) Memory Management and Setup**

Before launching the kernels, the host allocates the necessary Device Memory (VRAM) for the pipeline. This includes the input volume, intermediate volumes (for thresholded and blurred data), the output 2D image, the $3\times3$ rotation matrix, and the 27-element Gaussian kernel. To optimize GPU performance, the $3\times3\times3$ Gaussian kernel weights are precomputed on the Host (CPU) since they only depend on the `--sigma` parameter. The calculated weights are then transferred to the device memory, avoiding redundant mathematical computations across millions of GPU threads.

**2) Thresholding Kernel**

The first stage cleans the volume by removing low-intensity noise. Since this is a simple point operation, it was implemented using a 1D grid and block configuration (block = 256 threads).

Each thread calculates its global index (blockIdx.x * blockDim.x + threadIdx.x) and compares its assigned voxel against the threshold. If the voxel intensity is below the threshold, it is clamped to 0; otherwise, it retains its original value.

**3) 3D Gaussian Blur Kernel**

The second stage is a spatial convolution (stencil operation) that smooths the data to prevent single-pixel noise from dominating the final projection. This kernel is mapped to the 3D volume using a 3D block (8x8x4) and grid structure. Each thread is responsible for one voxel $(x, y, z)$. It iterates over the $3\times3\times3$ neighborhood, multiplying the neighboring voxel intensities by the precomputed weights transferred from the host.

To handle edge cases safely, the kernel uses max() and min() bounds clamping to ensure the convolution does not read out of the volume boundaries.

**4) Maximum Intensity Projection (MIP) Kernel**

The final stage reduces the 3D volume into a 2D image. Since the output is a 2D raster image ($512\times512$), the kernel uses a 2D block (16x16) and grid configuration. Each thread corresponds to a single pixel on the output screen. The algorithm simulates parallel rays fired through the volume by iterating over a pre-calculated ray_range. In each step, the thread calculates the 3D coordinates $(u, v, w)$ and uses the rotation matrix $R$ to perform backward mapping, transforming the ray coordinates into the rotated volume space. It bounds-checks the transformed coordinates, reads the voxel intensity, and keeps track of the maximum value encountered.

The maximum intensity is then written to the thread's respective pixel on the output image array.

**Innovation (Additional Features)**

A kernel that can calculate several MIP images was added. It is based on the already existing kernel for just one MIP image, but it uses a 3D block
since the variable amout of images can be interpreted as the third dimension. Each thread in the kernel corresponds to a single pixel in a single output image. 

The `main.cpp` program has been updated to call the new kernel if the `--steps` argument is set to more than 1. If `--steps` is set to more than 1, it animates the volume rotating from its starting rotation to its final rotation in `steps` frames. It saves each individual frame of the animation as its own image in the output/ folder and, if `ffmpeg` is installed on the host, it also saves the animation as a .mp4 video. 

## Analysis

The application was benchmarked using the Stanford Bunny CT scan dataset ($512\times512\times361$ volume) across various parameter configurations to evaluate the performance gains of the CUDA implementation.

### Comprehensive Benchmark Results

| Threshold | Sigma | Pitch | Yaw | Roll | CPU Time (ms) | GPU Time (ms) | Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|     32768 |   1.0 |     0 |   0 |    0 |       6862.19 |        838.19 |    8.19x |
|      8192 |   1.0 |     0 |   0 |    0 |       6669.60 |        826.79 |    8.07x |
|     16384 |   1.0 |     0 |   0 |    0 |       6977.69 |        840.12 |    8.31x |
|     49152 |   1.0 |     0 |   0 |    0 |       6948.30 |        832.54 |    8.35x |
|     32768 |   0.5 |     0 |   0 |    0 |       6687.83 |        829.32 |    8.06x |
|     32768 |   2.0 |     0 |   0 |    0 |       6642.08 |        831.54 |    7.99x |
|     32768 |   3.0 |     0 |   0 |    0 |       6852.14 |        844.32 |    8.12x |
|     32768 |   1.0 |    45 |   0 |    0 |       4773.57 |        833.46 |    5.73x |
|     32768 |   1.0 |     0 |  45 |    0 |       5684.42 |        837.86 |    6.78x |
|     32768 |   1.0 |    45 |  45 |   45 |       6132.30 |        848.44 |    7.23x |
|      8192 |   2.5 |    30 |  60 |   90 |       5568.17 |        839.07 |    6.64x |

### Speedup and Performance Discussion

- **Overall Speedup:** The overall speedup sits between 6x and 8x depending on the options. 

- **Impact of Thresholding:** The treshold does not seem to affect the speedup or the execution time, which makes sense since the number of comparisons is not being affected, only the number being compared against is being affected. 

- **Impact of Gaussian Blur (Sigma):** Similarly to the thresholding, the Gaussian Blur does not affect speedup or execution time since only the content, rather than the amount, of operations is being affected.

- **Impact of Rotation (Ray-Casting Complexity):** Changing the rotation away from (0,0,0) negatively affects speedup, but does not seem to negatively affect GPU execution time. In fact, it seems to speed up CPU execution time.
