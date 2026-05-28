
#include <cmath>
#include <cstdint>
#include <cstring>

#include "print.h"
#include "bunnyIO.h"
#include "imageIO.h"

#include "bunnyMIP.h"
#include "device-bunnyMIP.h"

inline float d2r(float angle) { return angle * M_PI / 180.0; }

void generate_rotation_matrix(float pitch, float yaw, float roll, float* R, bool inverse = false) {
    float cp = std::cos(pitch); float sp = std::sin(pitch);
    float cy = std::cos(yaw);   float sy = std::sin(yaw);
    float cr = std::cos(roll);  float sr = std::sin(roll);

    // Combined rotation R = Rz * Ry * Rx
    float m[3][3];
    m[0][0] = cy * cr;
    m[0][1] = cy * sr;
    m[0][2] = -sy;

    m[1][0] = sp * sy * cr - cp * sr;
    m[1][1] = sp * sy * sr + cp * cr;
    m[1][2] = sp * cy;

    m[2][0] = cp * sy * cr + sp * sr;
    m[2][1] = cp * sy * sr - sp * cr;
    m[2][2] = cp * cy;

    // Fill the flattened array
    if (!inverse) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[i * 3 + j] = m[i][j];
    } else {
        // Transpose for backward mapping (Inverse rotation)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[j * 3 + i] = m[i][j];
    }
}



int main(int argc, char* argv[]) {
    print("CLE2026 - BunnyMIP\n");

    uint16_t threshold = 1 << 15;
    float sigma = 1.0;
    float pitch = 0.0, yaw = 0.0, roll = 0.0;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            threshold = (uint16_t)std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            sigma = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--roll") == 0 && i + 1 < argc) {
            roll = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--pitch") == 0 && i + 1 < argc) {
            pitch = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--yaw") == 0 && i + 1 < argc) {
            yaw = std::atof(argv[++i]);
        }
    }

    print("  threshold=%u sigma=%.2f pitch=%.1f yaw=%.1f roll=%.1f\n",
          threshold, sigma, pitch, yaw, roll);

    uint16_t* volume = loadBunnyCT("data");

    // Raster output when running on the host
    uint16_t* h_raster = new uint16_t[kBunnySize*kBunnySize];
    // Raster output when running on the GPU
    uint16_t* d_raster = new uint16_t[kBunnySize*kBunnySize];

    float R[3*3];
    generate_rotation_matrix(d2r(pitch), d2r(yaw), d2r(roll), R);

    // CPU
    host_bunny_mip(volume, threshold, sigma, R, h_raster);
    // GPU (CUDA)
    device_bunny_mip(volume, threshold, sigma , R, d_raster);

    int raster_size = kBunnySize * kBunnySize;
    int diff = 0;
    for (int i = 0; i < raster_size; i++) {
        int d = abs((int)h_raster[i] - (int)d_raster[i]);
        if (d > 2)
            diff++;
    }

    float pct = (diff / (float)raster_size) * 100.0f;
    print("\n>> Output difference: %.2f%%\n", pct);
    print("VALIDATION: diff_pct=%.4f threshold=2 pixels_exceeding=%d total=%d\n",
          pct, diff, raster_size);

    savePGM16("output/bunnyMIP_cpu.pgm", h_raster, kBunnySize, kBunnySize);
    savePGM16("output/bunnyMIP_gpu.pgm", d_raster, kBunnySize, kBunnySize);

    delete [] volume;
    delete [] h_raster;
    delete [] d_raster;
    return (pct > 1.0f) ? 1 : 0;
}
