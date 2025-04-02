#include <executorch/kernels/portable/NativeFunctions.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

using namespace ::testing;
using Tensor = exec_aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using SizesType = executorch::aten::SizesType;
using TensorFactory = torch::executor::testing::TensorFactory;

class OpGridSampler2DOutKernelTest : public OperatorTest {
 protected:
  Tensor& grid_sampler_2d_out(
    const Tensor& input, 
    const Tensor& grid, 
    int64_t interpolation_mode,
    int64_t padding_mode, 
    bool align_corners, Tensor& out) {
    return torch::executor::native::grid_sampler_2d_out(context_, input, grid, interpolation_mode, padding_mode, align_corners, out);
  }
};

TEST_F(OpGridSampler2DOutKernelTest, BasicFunctionality) {

  // Create input tensor (N, C, H, W)
  Tensor input = TensorFactory::from_data<size_t>({1, 3, 4, 4}, {
    1,  2,  3,  4,   5,  6,  7,  8,   9, 10, 11, 12,  13, 14, 15, 16,
    17, 18, 19, 20,  21, 22, 23, 24,  25, 26, 27, 28,  29, 30, 31, 32,
    33, 34, 35, 36,  37, 38, 39, 40,  41, 42, 43, 44,  45, 46, 47, 48
  });

  // Create grid tensor (N, H_out, W_out, 2)
  Tensor grid = TensorFactory::from_data<size_t>({1, 4, 4, 2}, {
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
  });
  
  // Set the output tensor size
  Tensor out = TensorFactory::create<size_t>({1, 3, 4, 4}); // Same shape as input

  // Call the grid_sampler_2d_out function
  grid_sampler_2d_out(input, grid, 0, 0, false, out); // Bilinear, zero padding, no align_corners

  // Expected output data
  size_t expected_out_data[48] = {
    6, 6, 6, 6,   6, 6, 6, 6,   6, 6, 6, 6,   6, 6, 6, 6,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30
  };

  // Compare output tensor with expected values
  float* out_ptr = out.data_ptr<size_t>();
  for (int i = 0; i < 48; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], expected_out_data[i]);
  }

}
