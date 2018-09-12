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
  void operator()(const CPUDevice& d,
                  const float * ref,
                  int           ref_nb,
                  const float * query,
                  int           query_nb,
                  int           dim,
                  int           k,
                  float *       knn_dist,
                  int *         knn_index) {
    throw std::runtime_error("Unimplemented.");
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
    // grab input_ref tensors
    const Tensor& tensor_ref = context->input(0);
    const Tensor& tensor_query = context->input(1);
    // [n, d]
    auto input_ref = tensor_ref.tensor<float, 2>();
    auto input_query = tensor_query.tensor<float, 2>();

    // n, k, d
    const auto np = input_ref.dimension(0);
    const auto n = input_query.dimension(0);
    const auto k = context->input(2).flat<std::int32_t>()(0);
    const auto d = input_query.dimension(1);

    // *******************************************************
    // Create output tensor
    // *******************************************************
    Tensor *ot_indices = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(
                    0, TensorShape{{n, k}},
                    &ot_indices));
    Tensor *ot_distances = nullptr;
    OP_REQUIRES_OK(
            context,
            context->allocate_output(
                    1, TensorShape{{n, k}},
                    &ot_distances));


    auto output_indices = ot_indices->tensor<std::int32_t, 2>();
    auto output_distances = ot_distances->tensor<float, 2>();

    // *******************************************************
    // Do the computation.
    // *******************************************************
    KNNFunctor<Device>()(
            context->eigen_device<Device>(),
            input_ref.data(),
            static_cast<int>(np),
            input_query.data(),
            static_cast<int>(n),
            static_cast<int>(d),
            static_cast<int>(k),
            output_distances.data(),
            output_indices.data());
  }
};

Status ResizeShapeFn(::tensorflow::shape_inference::InferenceContext* c) {
  using namespace ::tensorflow::shape_inference;
  ShapeHandle shape_ref;
  ShapeHandle shape_query;
  DimensionHandle dim;
  // check ref tensor == {n, dim}
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shape_ref));
  // check query tensor == {n, dim}
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &shape_query));
  // check equal dimension dimension of ref and query
  const auto dim_or_const_ref = DimensionOrConstant(c->Dim(shape_ref, 1));
  const auto dim_or_const_query = DimensionOrConstant(c->Dim(shape_query, 1));
  if (InferenceContext::ValueKnown(dim_or_const_ref) && InferenceContext::ValueKnown(dim_or_const_query)
          && InferenceContext::Value(dim_or_const_ref) != InferenceContext::Value(dim_or_const_query))
  {
    return Status(error::Code::FAILED_PRECONDITION,
                  "Dimension of ref and query points does not match.");
  }
  // check k input tensor (=scalar)
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &shape_ref));

  // get number of query points
  std::int64_t n = InferenceContext::kUnknownDim;
  const auto dim_or_const_query_n = DimensionOrConstant(c->Dim(shape_query, 0));
  if (InferenceContext::ValueKnown(dim_or_const_query_n))
    n = InferenceContext::Value(dim_or_const_query_n);

  // 0: indices, 1: distances. [n_query, k]
  c->set_output(0, c->MakeShape({n, InferenceContext::kUnknownDim}));
  c->set_output(1, c->MakeShape({n, InferenceContext::kUnknownDim}));

  return Status::OK();
}

//
REGISTER_OP("KNearestNeighbor")
    // batched (padded) point clouds
    .Input("ref: float32")
    .Input("query: float32")
    .Input("k: int32")
    .Output("indices: int32")
    .Output("distances: float32")
    .SetShapeFn(ResizeShapeFn);


// Register the CPU kernels.
//REGISTER_KERNEL_BUILDER(
//    Name("KNearestNeighbor")
//    .Device(DEVICE_CPU),
//    KNearestNeighborOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
/* Declare explicit instantiations in kernel_example.cu.cc. */
extern template struct KNNFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(
    Name("KNearestNeighbor").Device(DEVICE_GPU),
    KNearestNeighborOp<GPUDevice>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
