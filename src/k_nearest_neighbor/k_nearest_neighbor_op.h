// 2018, Patrick Wieschollek <mail@patwie.com>

#ifndef K_NEAREST_NEIGHBOR_KERNELS_K_NEAREST_NEIGHBOR_OP_H_
#define K_NEAREST_NEIGHBOR_KERNELS_K_NEAREST_NEIGHBOR_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename Dtype>
struct KNNFunctor {
  static void launch(::tensorflow::OpKernelContext* context,
                     const Tensor& t_ref,
                     const Tensor& t_query,
                     Tensor* distances,
                     Tensor* indices,
                     std::int32_t k);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // K_NEAREST_NEIGHBOR_KERNELS_MATRIX_ADD_OP_H_
