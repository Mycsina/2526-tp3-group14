
#pragma once

#include <cstdint>

void device_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output);

void device_bunny_multiple_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output, uint16_t num_images);

