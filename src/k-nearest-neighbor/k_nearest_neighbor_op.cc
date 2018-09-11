// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
// -- END LICENSE BLOCK ------------------------------------------------
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Christoph Rist <c.rist@posteo.de>
 * \date    2018-09-11
 *
 */
//----------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <functional>
#include <unordered_map>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include "k_nearest_neighbor_op.h"


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <>
struct KNNFunctor<CPUDevice> {
  void operator()(const CPUDevice& d, int size, const float* in, std::int32_t* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

namespace tensorflow {

template <typename Device>
class KNearestNeighborOp : public OpKernel
{
public:

  explicit KNearestNeighborOp(OpKernelConstruction* context) : OpKernel(context)
  {}

  void Compute(OpKernelContext* context) override
  {
    // k
    const auto k = context->input(1).flat<std::int32_t>()(0);

    // grab input point cloud tensor
    const Tensor& tensor_point_cloud = context->input(0);
    // [batch, n points, xyz]
    auto input = tensor_point_cloud.tensor<float, 3>();
    const auto batch_size = input.dimension(0);
    const auto max_num_points = input.dimension(1);

    // *******************************************************
    // Create output tensor
    // *******************************************************
    Tensor *ot_indices = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(
                    0, TensorShape{{batch_size, max_num_points, k}},
                    &ot_indices));
    auto tensor_indices = ot_indices->tensor<std::int32_t, 3>();

    // *******************************************************
    // Do the computation.
    // *******************************************************
    KNNFunctor<Device>()(
            context->eigen_device<Device>(),
            static_cast<int>(tensor_point_cloud.NumElements()),
            input.data(),
            tensor_indices.data());
  }
};

Status ResizeShapeFn(::tensorflow::shape_inference::InferenceContext* c) {
  using namespace ::tensorflow::shape_inference;
  ShapeHandle shape;
  DimensionHandle dim;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &shape));
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(shape, 2), 3, &dim));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &shape));
  c->set_output(0, c->MakeShape({InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim,
                                 InferenceContext::kUnknownDim}));
  return Status::OK();
}

//
REGISTER_OP("KNearestNeighbor")
    // batched (padded) point clouds
    .Input("point_cloud: float32")
    .Input("k: int32")
    .Output("indices: int32")
    .SetShapeFn(ResizeShapeFn);

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(
    Name("KNearestNeighbor")
    .Device(DEVICE_CPU),
    KNearestNeighborOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
/* Declare explicit instantiations in kernel_example.cu.cc. */
extern template ExampleFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(
    Name("Example").Device(DEVICE_GPU),
    KNearestNeighborOp<GPUDevice>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
