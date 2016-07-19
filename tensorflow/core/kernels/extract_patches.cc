// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

  struct ExtractPatchesState {
    // ValidateAndCreateOutput checks the bounds on the input tensors
    // and requested size.
    // If any of these operations fails, it sets an error status in
    // the context, which the caller must check.
    void ValidateAndCreateOutput(OpKernelContext* context) {
      const auto& input = context->input(0);
      const auto& input_shape = input.shape();
      const auto& num_dims = input_shape.dims();

      OP_REQUIRES(
          context, num_dims == 4,
          errors::InvalidArgument(
              "input must be 4-dimensional (batch_size, height, width, depth)",
              input_shape.DebugString()));

      batch_size = input_shape.dim_size(0);
      input_height = input_shape.dim_size(1);
      input_width = input_shape.dim_size(2);
      depth = input_shape.dim_size(3);

      const auto& window_size = context->input(1);
      OP_REQUIRES(context, (window_size.shape().dims() == 1) &&
                               window_size.shape().dim_size(0) == 2,
                  errors::InvalidArgument(
                      "patch shape must be a vector of size 2 (height, width)",
                      window_size.shape().DebugString()));

      patch_height = window_size.tensor<int, 1>()(0);
      patch_width = window_size.tensor<int, 1>()(1);

      const auto& offsets = context->input(2);
      OP_REQUIRES(context, offsets.shape().dims() == 3,
                  errors::InvalidArgument("input must be a tensor [batch_size, num_patches, 2]",
                                          offsets.shape().DebugString()));
      OP_REQUIRES(context, offsets.shape().dim_size(0) == batch_size,
                  errors::InvalidArgument("first dimension should be batch",
                                          offsets.shape().DebugString()));
      OP_REQUIRES(
          context, offsets.shape().dim_size(2) == 2,
          errors::InvalidArgument("third dimension should be of size 2 (y,x)",
                                  offsets.shape().DebugString()));

      num_patches = offsets.shape().dim_size(1);
      TensorShape output_shape({batch_size, num_patches, patch_height, patch_width, depth});

      output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
      if (output->NumElements() == 0) {
        // Nothing else to do.
        return;
      }
    }
    int32 batch_size;
    int32 input_height;
    int32 input_width;
    int32 depth;
    int32 patch_height;
    int32 patch_width;
    int32 num_patches;
    Tensor* output;
  };

class ExtractPatchesOp : public OpKernel {
 public:
  explicit ExtractPatchesOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  // Expect input tensor of rank 4 with dimensions (batch_size, height, width,
  // depth).
  void Compute(OpKernelContext* context) override {
    ExtractPatchesState st;
    auto& input = context->input(0);
    auto& offsets = context->input(2);
    st.ValidateAndCreateOutput(context);

    typename TTypes<float, 5>::Tensor output_patches = st.output->tensor<float, 5>();
    typename TTypes<float, 4>::ConstTensor input_images = input.tensor<float, 4>();

    for (int i = 0; i < st.batch_size; ++i) {
      for (int n = 0; n < st.num_patches; ++n) {
        const float offset_y = offsets.tensor<float, 3>()(i, n, 0);
        const float offset_x = offsets.tensor<float, 3>()(i, n, 1);

        for(int source_x=offset_x - st.patch_width/2, target_x=0;
                target_x < st.patch_width;
                ++source_x, ++target_x) {
          for(int source_y=offset_y - st.patch_height/2, target_y=0;
                target_y < st.patch_height;
                ++source_y, ++target_y) {
            if (source_x > 0 && source_x < st.input_width && source_y > 0 && source_y < st.input_height) {
              for (int c = 0; c < st.depth; ++c) {
                output_patches(i, n, target_y, target_x, c) = input_images(i, source_y, source_x, c);
              }
            }
          }
        }
      }
    }

  }

};

REGISTER_KERNEL_BUILDER(Name("ExtractPatches").Device(DEVICE_CPU),
                        ExtractPatchesOp);


#if GOOGLE_CUDA

template <typename T>
class ExtractPatchesGPUOp : public OpKernel {
public:
explicit ExtractPatchesGPUOp(OpKernelConstruction* context)
  : OpKernel(context) { }

void Compute(OpKernelContext* context) override {
  ExtractPatchesState st;
  auto& input = context->input(0);
  auto& offsets = context->input(2);
  st.ValidateAndCreateOutput(context);

  bool status = ExtractPatches<T>(
      st.input->flat<T>().data(), st.batch_size, st.input_height, st.input_width,
      st.depth, st.patch_height, st.patch_width, st.offsets, st.num_patches,
      st.output->flat<T>().data(),
      context->eigen_gpu_device());

  if (!status) {
    context->SetStatus(
        errors::Internal("Failed launching ExtractPatchesGPUOp"));
  }
}
}

#define REGISTER_KERNEL(T)                                        \
REGISTER_KERNEL_BUILDER(Name("ExtractPatches")           \
                          .Device(DEVICE_GPU)                 \
                          .TypeConstraint<T>("T")             \
                          .HostMemory("size"),                \
                          ExtractPatchesGPUOp<T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
#endif

}  // end namespace tensorflow
