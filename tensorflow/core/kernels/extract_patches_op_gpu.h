#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_EXTRACT_PATCHES_OP_GPU_H_
#define TENSORFLOW_CORE_KERNELS_EXTRACT_PATCHES_OP_GPU_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <typename T>
bool ExtractPatches(const T* bottom_data, const int batch, const int in_height,
                    const int in_width, const int channels, const int patch_height,
                    const int patch_width, int* offsets, int num_offsets
                    T* top_data, const Eigen::GpuDevice& d);


}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_EXTRACT_PATCHES_OP_GPU_H_
