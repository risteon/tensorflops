// -- BEGIN LICENSE BLOCK ----------------------------------------------
// Creative Commons License
//
// Original work Copyright 2018 Vincent Garcia, Éric Debreuve, Michel Barlaud
// Modified work Copyright 2018 Christoph Rist
//
// Attribution-Noncommercial-Share Alike 3.0 Unported
//
// You are free:
// * To Share: to copy, distribute and transmit the work
// * To Remix: to adapt the work
//
//Under the following conditions:
//
// Attribution: You must attribute the work in the manner specified by the author or licensor
//   (but not in any way that suggests that they endorse you or your use of the work).
//
// Noncommercial: You may not use this work for commercial purposes.
//
// Share Alike: If you alter, transform, or build upon this work, you may distribute the resulting
//   work only under the same or similar license to this one.
//
// For more information, please consult the page http://creativecommons.org/licenses/by-nc-sa/3.0/
// -- END LICENSE BLOCK ------------------------------------------------

#ifndef K_NEAREST_NEIGHBOR_OP_H
#define K_NEAREST_NEIGHBOR_OP_H

template <typename Device>
struct KNNFunctor {
  void operator()(const Device& d,
                  const float * ref,
                  int           ref_nb,
                  const float * query,
                  int           query_nb,
                  int           dim,
                  int           k,
                  float *       knn_dist,
                  int *         knn_index);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <>
struct KNNFunctor<Eigen::GpuDevice> {
  void operator()(const Eigen::GpuDevice& d,
                  const float * ref,
                  int           ref_nb,
                  const float * query,
                  int           query_nb,
                  int           dim,
                  int           k,
                  float *       knn_dist,
                  int *         knn_index);
};
#endif

bool knn_cuda_tf(const float * ref,
                 int           ref_nb,
                 const float * query,
                 int           query_nb,
                 int           dim,
                 int           k,
                 float *       knn_dist,
                 int *         knn_index);


/**
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * This implementation uses global memory to store reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
bool knn_cuda_global(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,
                     int *         knn_index);


/**
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * This implementation uses texture memory for storing reference points  and memory to store query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
bool knn_cuda_texture(const float * ref,
                      int           ref_nb,
                      const float * query,
                      int           query_nb,
                      int           dim,
                      int           k,
                      float *       knn_dist,
                      int *         knn_index);


# endif //< K_NEAREST_NEIGHBOR_OP_H
