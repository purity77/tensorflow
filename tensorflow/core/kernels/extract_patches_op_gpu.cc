#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/extract_patches_op_gpu.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {

template <typename T>
__global__ void ExtractPatchesNHWC(const int nthreads, const T* source_data,
                                  const int in_height, const int in_width,
                                  const int channels, const int patch_height,
                                  const int patch_width, int* offsets,
                                  const int num_offsets, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % patch_width;
    n /= patch_width;
    int out_y = n % patch_height;
    n /= patch_height;
    int offset = n % num_offsets;
    n /= num_offsets;

    const int offset_x = ldg(offsets + n * num_offsets);
    const int offset_y = ldg(offsets + n * num_offsets + 1);

    // Get the n_th image.
    const T* source_data_n = source_data + n * channels * in_height * in_width;

    // Compute the position on the output image offset by the patch location.
    out_x = static_cast<int>(floorf(out_x + offset_x - patch_width / 2));
    out_y = static_cast<int>(floorf(out_y + offset_y - patch_height / 2));

    // Clip if out of boundaries.
    const int in_x = min(max(out_x, 0), in_width - 1);
    const int in_y = min(max(out_y, 0), in_height - 1);

    // Computes the offset for the source pixel.
    const int source_offset = (in_y * in_width + in_x) * channels + c;

    // Sets the source pixel to the target position.
    top_data[index] = ldg(source_data_n + source_offset);
  }
}

}  // namespace

template <typename T>
bool ExtractPatches(const T* bottom_data, const int batch, const int in_height,
                    const int in_width, const int channels, const int patch_height,
                    const int patch_width, int* offsets, int num_offsets
                    T* top_data, const Eigen::GpuDevice& d){
  const int output_size = batch * channels * patch_height * patch_width * num_offsets;
  CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);

  ExtractPatchesNHWC<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      output_size, bottom_data, in_height, in_width, channels, patch_height,
      patch_width, offsets, num_offsets, top_data);
  return d.ok();
}

#define DECLARE_GPU_SPEC(T)                                                        \
  template bool ExtractPatchesNHWC(const T* bottom_data, const int batch,       \
                               const int in_height, const int in_width,            \
                               const int channels,const int patch_height, \
                               const int patch_width, int* offsets, int num_offsets, \
                               T* top_data,               \
                               const Eigen::GpuDevice& d);

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
