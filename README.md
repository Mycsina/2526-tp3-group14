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

...

**1) Memory Management and Setup**

...

**2) Thresholding Kernel**

...

**3) 3D Gaussian Blur Kernel**

...

**4) Maximum Intensity Projection (MIP) Kernel**

...

**Innovation (Additional Features)**

[TODO]

## Analysis

[TODO]
