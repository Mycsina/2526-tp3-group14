# CLE - Practical Assignment 3

**Volumetric Maximum Intensity Projection (MIP) with CUDA Programming**

*Group 14*

[108269] André Cardoso

[112793] Cláudia Seabra

[135716] Viktor Bogojevic

## Instructions

Compiling the application: make

Running the application: `./bunnyMIP [--threshold ] [--sigma ] [--roll ] [--pitch ] [--yaw ]`

- `--threshold`: Defines the intensity floor. Any voxel below this value is set to 0 (default: 32768).
- `--sigma`: Defines the standard deviation ($\sigma$) used to calculate the Gaussian weights for the blur filter (default: 1.0).
- `--roll`, `--pitch`, `--yaw`: Defines the rotation of the volume in degrees (default: 0.0 for all).

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

[TODO]

## Analysis

The application was benchmarked using the Stanford Bunny CT scan dataset ($512\times512\times361$ volume) across various parameter configurations to evaluate the performance gains of the CUDA implementation.

### Comprehensive Benchmark Results

| Threshold | Sigma | Pitch | Yaw | Roll | CPU Time (ms) | GPU Time (ms) | Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32768 | 1.0 | 0 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 8192 | 1.0 | 0 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 16384 | 1.0 | 0 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 49152 | 1.0 | 0 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 32768 | 0.5 | 0 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 32768 | 2.0 | 0 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 32768 | 3.0 | 0 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 32768 | 1.0 | 45 | 0 | 0 | [TBD] | [TBD] | [TBD]x |
| 32768 | 1.0 | 0 | 45 | 0 | [TBD] | [TBD] | [TBD]x |
| 32768 | 1.0 | 45 | 45 | 45 | [TBD] | [TBD] | [TBD]x |
| 8192 | 2.5 | 30 | 60 | 90 | [TBD] | [TBD] | [TBD]x |

### Speedup and Performance Discussion

- **Overall Speedup:** ...

- **Impact of Thresholding:** ...

- **Impact of Gaussian Blur (Sigma):** ...

- **Impact of Rotation (Ray-Casting Complexity):** ...

- **Memory Transfer Overhead:** ...
