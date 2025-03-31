// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
 
namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

template <typename CTYPE>
void grid_sampler_2d_wrapper(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {

    const int N = input.size(0);
    const int C = input.size(1);
    const int iH = input.size(2);
    const int iW = input.size(3);
    const int oH = grid.size(1);
    const int oW = grid.size(2);

    // Access input and output raw data pointers
    const CTYPE* input_data = input.data_ptr<CTYPE>();
    CTYPE* out_data = out.data_ptr<CTYPE>();

    // Converts normalized coordinates in the range [-1, 1] into image coordinates in the range [0, size - 1]
    auto unnormalize = [&](float coord, int size) -> float {
        float scale = (align_corners ? size * 0.5 - 0.5 : size * 0.5);
        return coord * scale + (size * 0.5 - 0.5);
    };

    // Ensures out-of-bounds coordinates are reflected back into the valid range
    auto reflect_coordinates = [&](float coord, int low, int high) -> int {
        if (low == high) return 0;
        float span = (high - low) / 2;
        float coords2 = std::abs(coord - low / 2);
        float extra = std::fmod(coords2, span);
        int flips = static_cast<int>(coords2 / span);
        return (flips % 2 == 0) ? extra + low / 2 : span + low / 2 - extra;
    };

    // Determines the source index in the image
    auto compute_source_index = [&](float coord, int size) -> int {
        float unnormalized = unnormalize(coord, size);
        if (padding_mode == 0) { // Zero padding
            return std::clamp(static_cast<int>(std::round(unnormalized)), 0, size - 1);
        } else if (padding_mode == 1) { // Border padding
            return std::clamp(static_cast<int>(unnormalized), 0, size - 1);
        } else { // Reflection padding
            if (align_corners) {
                return reflect_coordinates(unnormalized, 0, 2 * (size - 1));
            } else {
                return reflect_coordinates(unnormalized, -1, 2 * size - 1);
            }
        }
    };

    // Iterate through output tensor
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < oH; ++h) {
            for (int w = 0; w < oW; ++w) {
                // Access grid raw data pointer
                const float* grid_data = grid.data_ptr<float>();

                // Retrieve grid sampling coordinates
                float x = grid_data[n * oH * oW * 2 + h * oW * 2 + w * 2 + 0];
                float y = grid_data[n * oH * oW * 2 + h * oW * 2 + w * 2 + 1];

                // Compute source indices
                int ix = compute_source_index(x, iW);
                int iy = compute_source_index(y, iH);

                if (interpolation_mode == 0) { // Bilinear interpolation

                    // 4 surrounding pixels indices with floor and ceil
                    int ix_nw = std::floor(ix);
                    int iy_nw = std::floor(iy);
                    int ix_ne = ix_nw + 1;
                    int iy_sw = iy_nw + 1;

                    // Distances of surrounding pixels from interpolated value (weights)
                    float w_nw = (ix_ne - ix) * (iy_sw - iy);
                    float w_ne = (ix - ix_nw) * (iy_sw - iy);
                    float w_sw = (ix_ne - ix) * (iy - iy_nw);
                    float w_se = (ix - ix_nw) * (iy - iy_nw);

                    // Weighted sum
                    out_data[n * C * oH * oW + C * h * oW + w] = 
                        input_data[n * C * iH * iW + C * iy_nw * iW + ix_nw] * w_nw +
                        input_data[n * C * iH * iW + C * iy_nw * iW + ix_ne] * w_ne +
                        input_data[n * C * iH * iW + C * iy_sw * iW + ix_nw] * w_sw +
                        input_data[n * C * iH * iW + C * iy_sw * iW + ix_ne] * w_se;

                } else if (interpolation_mode == 1) { // Nearest interpolation

                    // Round to nearest pixel index
                    int ix_nearest = std::round(ix);
                    int iy_nearest = std::round(iy);

                    // Assign nearest pixel value
                    out_data[n * C * oH * oW + C * h * oW + w] = input_data[n * C * iH * iW + C * iy_nearest * iW + ix_nearest];

                } else if (interpolation_mode == 2) { // Bicubic interpolation

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

    const int N = input.size(0);
    const int C = input.size(1);
    const int iH = input.size(2);
    const int iW = input.size(3);
    const int oH = grid.size(1);
    const int oW = grid.size(2);

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
