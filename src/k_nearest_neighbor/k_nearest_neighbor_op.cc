// 2018, Patrick Wieschollek <mail@patwie.com>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "k_nearest_neighbor_op.h"

namespace tensorflow {

template <typename Device, typename Dtype>
class KNearestNeighborOp : public OpKernel {
 public:
  explicit KNearestNeighborOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
  }

  void Compute(OpKernelContext* context) override {
    // grab input_ref tensors
    const Tensor& tensor_ref = context->input(0);
    const Tensor& tensor_query = context->input(1);
    // [n, d]
    auto input_ref = tensor_ref.tensor<Dtype, 2>();
    auto input_query = tensor_query.tensor<Dtype, 2>();

    if (!context->status().ok()) {
      return;
    }

    // n, k, d
    const std::int32_t np = input_ref.dimension(1);
    const std::int32_t n = input_query.dimension(1);
    const std::int32_t d = input_query.dimension(0);

    // Create output tensor
    Tensor *ot_indices = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(
                    0, TensorShape{{n, k_}},
                    &ot_indices));
    Tensor *ot_distances = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(
                    1, TensorShape{{n, k_}},
                    &ot_distances));

    // call kernel
    ::tensorflow::functor::KNNFunctor<Device, Dtype>::launch(context,
                                                             tensor_ref, tensor_query,
                                                             ot_distances, ot_indices,
                                                             k_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(KNearestNeighborOp);

  std::int32_t k_;
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T)                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), \
      NAME##Op<DEVICE##Device, T>)


#ifdef GOOGLE_CUDA
REGISTER_CUSTOM_OP(KNearestNeighbor, GPU, float);
#endif  // GOOGLE_CUDA
#undef REGISTER_CUSTOM_OP

}  // namespace tensorflow
