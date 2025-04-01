#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
 
namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using SizesType = executorch::aten::SizesType;

//
// Helper Functions
//

// Converts normalized coordinates in the range [-1, 1] into image coordinates in the range [0, size - 1]
inline size_t grid_sampler_unnormalize(size_t coord, int64_t size, bool align_corners) {
    if (align_corners) {
        // Unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1);
    } else {
        // Unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1) * size - 1) / 2;
    }
}

// Clips coordinates to between 0 and clip_limit - 1
inline size_t clip_coordinates(size_t coord, int64_t clip_limit) {
    return std::min(static_cast<size_t>(clip_limit - 1), std::max(coord, static_cast<size_t>(0)));
}

// Ensures out-of-bounds coordinates are reflected back into the valid range
inline size_t reflect_coordinates(size_t coord, int64_t twice_low, int64_t twice_high) {
    if (twice_low == twice_high) {
        return 0;
    }
    size_t min = twice_low / 2;
    size_t span = (twice_high - twice_low) / 2;
    coord = std::fabs(coord - min);
    size_t extra = std::fmod(coord, span);
    int64_t flips = static_cast<int64_t>(std::floor(coord / span));
    if (flips % 2 == 0) {
        return extra + min;
    } else {
        return span - extra + min;
    }
}

// Determines the source index in the image
inline size_t compute_coordinates(size_t coord, int64_t size, int64_t padding_mode, bool align_corners) {
    if (padding_mode == 0) { // Zeros
        // TBD
    } else if (padding_mode == 1) { // Border
        // Clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    } else { // Reflection
        // Reflect coordinates by image borders
        if (align_corners) {
        coord = reflect_coordinates(coord, 0, 2*(size - 1));
        } else {
        coord = reflect_coordinates(coord, -1, 2*size - 1);
        }
        // Clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }
    return coord;
}

template <typename CTYPE>
void grid_sampler_2d_wrapper(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t iH = input.size(2);
    const int64_t iW = input.size(3);
    const int64_t oH = grid.size(1);
    const int64_t oW = grid.size(2);

    // Access input and output raw data pointers
    const size_t* input_data = input.data_ptr<size_t>();
    size_t* out_data = out.data_ptr<size_t>();

    // Iterate through output tensor
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < oH; ++h) {
            for (int w = 0; w < oW; ++w) {
                // Access grid raw data pointer
                const size_t* grid_data = grid.data_ptr<size_t>();

                // Retrieve grid sampling coordinates
                size_t x = grid_data[n * oH * oW * 2 + h * oW * 2 + w * 2 + 0];
                size_t y = grid_data[n * oH * oW * 2 + h * oW * 2 + w * 2 + 1];

                // Compute source indices
                int ix = compute_coordinates(x, iW, padding_mode, align_corners);
                int iy = compute_coordinates(y, iH, padding_mode, align_corners);

                if (interpolation_mode == 0) { // Bilinear

                    // Get NE, NW, SE, SW pixel values from (x, y)
                    int64_t ix_nw = static_cast<int64_t>(::floor(ix));
                    int64_t iy_nw = static_cast<int64_t>(::floor(iy));
                    int64_t ix_ne = ix_nw + 1;
                    int64_t iy_ne = iy_nw;
                    int64_t ix_sw = ix_nw;
                    int64_t iy_sw = iy_nw + 1;
                    int64_t ix_se = ix_nw + 1;
                    int64_t iy_se = iy_nw + 1;

                    // Distances of surrounding pixels from interpolated value (weights)
                    size_t w_nw = (ix_se - ix)    * (iy_se - iy);
                    size_t w_ne = (ix    - ix_sw) * (iy_sw - iy);
                    size_t w_sw = (ix_ne - ix)    * (iy    - iy_ne);
                    size_t w_se = (ix    - ix_nw) * (iy    - iy_nw);

                    // Assign weighted sum
                    out_data[n * C * oH * oW + C * h * oW + w] = 
                        input_data[n * C * iH * iW + C * iy_nw * iW + ix_nw] * w_nw +
                        input_data[n * C * iH * iW + C * iy_nw * iW + ix_ne] * w_ne +
                        input_data[n * C * iH * iW + C * iy_sw * iW + ix_nw] * w_sw +
                        input_data[n * C * iH * iW + C * iy_sw * iW + ix_ne] * w_se;

                } else if (interpolation_mode == 1) { // Nearest

                    // Round to nearest pixel index
                    int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
                    int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));

                    // Assign nearest pixel
                    out_data[n * C * oH * oW + C * h * oW + w] = input_data[n * C * iH * iW + C * iy_nearest * iW + ix_nearest];

                } else if (interpolation_mode == 2) { // Bicubic

                    ix = grid_sampler_unnormalize(x, iW, align_corners);
                    iy = grid_sampler_unnormalize(y, iH, align_corners);

                    size_t ix_nw = std::floor(ix);
                    size_t iy_nw = std::floor(iy);

                    const size_t tx = ix - ix_nw;
                    const size_t ty = iy - iy_nw;

                    // TBD
                    
                }
            }
        }
    }

}

Tensor& grid_sampler_2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t iH = input.size(2);
    const int64_t iW = input.size(3);
    const int64_t oH = grid.size(1);
    const int64_t oW = grid.size(2);

    // Check for 4D input and grid
    ET_KERNEL_CHECK(ctx, (input.dim() == 4), InvalidArgument, out);
    ET_KERNEL_CHECK(ctx, (grid.dim() == 4), InvalidArgument, out);

    // Check for output
    ET_KERNEL_CHECK(ctx, (out.size(0) == N && out.size(1) == C && out.size(2) == oH && out.size(3) == oW), InvalidArgument, out);

    // Static name for debugging/logging
    static constexpr const char name[] = "grid_sampler_2d.out";

    // Macro to switch over scalar types
    ET_SWITCH_REALH_TYPES(input.scalar_type(), ctx, name, CTYPE, [&]() {
        grid_sampler_2d_wrapper<CTYPE>(input, grid, interpolation_mode, padding_mode, align_corners, out);
    });

    return out;
}
 
} // namespace native
} // namespace executor
} // namespace torch
