
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>

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
    float final_pitch = 0.0, final_yaw = 0.0, final_roll = 0.0;
    int steps = 1;

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
        } else if (std::strcmp(argv[i], "--final_roll") == 0 && i + 1 < argc) {
            final_roll = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--final_pitch") == 0 && i + 1 < argc) {
            final_pitch = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--final_yaw") == 0 && i + 1 < argc) {
            final_yaw = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = std::atoi(argv[++i]);
        }
    }

    print("  threshold=%u sigma=%.2f pitch=%.1f yaw=%.1f roll=%.1f\n", threshold, sigma, pitch, yaw, roll);
    print("  final_pitch=%.1f final_yaw=%.1f final_roll=%.1f steps = %d\n", final_pitch, final_yaw, final_roll, steps);

    uint16_t* volume = loadBunnyCT("data");

    // Raster output when running on the host
    uint16_t* h_raster = new uint16_t[kBunnySize*kBunnySize];
    // Raster output when running on the GPU
    uint16_t* d_raster = new uint16_t[kBunnySize*kBunnySize];

    float R[3*3];
    generate_rotation_matrix(d2r(pitch), d2r(yaw), d2r(roll), R);

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    host_bunny_mip(volume, threshold, sigma, R, h_raster);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    
    // GPU (CUDA)
    auto start_gpu = std::chrono::high_resolution_clock::now();
    device_bunny_mip(volume, threshold, sigma , R, d_raster);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_ms = end_gpu - start_gpu;

    double speedup = cpu_ms.count() / gpu_ms.count();

    print("\n>> Performance Analysis:\n");
    print("  CPU Time: %.2f ms\n", cpu_ms.count());
    print("  GPU Time: %.2f ms\n", gpu_ms.count());
    print("  Speedup: %.2fx\n", speedup);

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

    if (steps > 1) {
        // Raster output when running on the GPU
        uint16_t* d_rasters = new uint16_t[kBunnySize*kBunnySize*steps];
        float Rs[3*3*steps];
        float pitch_step = (pitch + final_pitch) / (steps + 1.f);
        float yaw_step = (yaw + final_yaw) / (steps + 1.f);
        float roll_step = (roll + final_roll) / (steps + 1.f);

        for (int i = 0; i < steps; i++)
            generate_rotation_matrix(d2r(pitch + i * pitch_step), d2r(yaw + i * yaw_step), d2r(roll + i * roll_step), &Rs[3*3*i]);

        // GPU (CUDA) - batch MIP
        device_bunny_multiple_mip(volume, threshold, sigma , Rs, d_rasters, 36);

        for (int i = 0; i < steps; i++) {
            char filename[64];
            sprintf(filename, "output/bunnyMIP_gpu_frame%02d.pgm", i);
            savePGM16(filename, &d_rasters[kBunnySize*kBunnySize*i], kBunnySize, kBunnySize);
        }

        std::string ffmpeg_save_video = "ffmpeg -y -framerate 10 -i output/bunnyMIP_gpu_frame%02d.pgm -c:v libx264 -pix_fmt yuv420p output/bunny_video.mp4";
        int ret = std::system(ffmpeg_save_video.c_str());
        if (ret) printf("Saving video failed\n");

        delete [] d_rasters;
    }

    delete [] volume;
    delete [] h_raster;
    delete [] d_raster;
    return (pct > 1.0f) ? 1 : 0;
}
