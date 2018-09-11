// 2018, Patrick Wieschollek <mail@patwie.com>

#include <cstring>

#include "k_nearest_neighbor_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct KNNFunctor<CPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* context,
                     const Tensor& t_ref,
                     const Tensor& t_query,
                     Tensor* distances,
                     Tensor* indices,
                     std::int32_t k) {

  }
};

template struct KNNFunctor<CPUDevice, int32>;
template struct KNNFunctor<CPUDevice, uint32>;
template struct KNNFunctor<CPUDevice, float>;
template struct KNNFunctor<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow