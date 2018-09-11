// 2018, Patrick Wieschollek <mail@patwie.com>

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "k_nearest_neighbor_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define BLOCK_DIM 16

namespace tensorflow {
namespace {

}
namespace functor {




/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 */
__global__ void compute_distances(float * ref,
                                  int     ref_width,
                                  int     ref_pitch,
                                  float * query,
                                  int     query_width,
                                  int     query_pitch,
                                  int     height,
                                  float * dist) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height-1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
    }
}


/**
 * Computes the squared Euclidean distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the texture memory
 * @param ref_width    number of reference points
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 */
__global__ void compute_distance_texture(cudaTextureObject_t ref,
                                         int                 ref_width,
                                         float *             query,
                                         int                 query_width,
                                         int                 query_pitch,
                                         int                 height,
                                         float*              dist) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ( xIndex<query_width && yIndex<ref_width) {
        float ssd = 0.f;
        for (int i=0; i<height; i++) {
            float tmp  = tex2D<float>(ref, (float)yIndex, (float)i) - query[i * query_pitch + xIndex];
            ssd += tmp * tmp;
        }
        dist[yIndex * query_pitch + xIndex] = ssd;
    }
}


/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k){

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i=1; i<height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i*dist_pitch];
            int   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index;
        }
    }
}


/**
 * Computes the square root of the first k lines of the distance matrix.
 *
 * @param dist   distance matrix
 * @param width  width of the distance matrix
 * @param pitch  pitch of the distance matrix given in number of columns
 * @param k      number of values to consider
 */
__global__ void compute_sqrt(float * dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}


/**
 * Computes the squared norm of each column of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    output array containing the squared norm values
 */
__global__ void compute_squared_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex<width){
        float sum = 0.f;
        for (int i=0; i<height; i++){
            float val = array[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}


/**
 * Add the reference points norm (column vector) to each colum of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    reference points norm stored as a column vector
 */
__global__ void add_reference_points_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int xIndex = blockIdx.x * blockDim.x + tx;
    unsigned int yIndex = blockIdx.y * blockDim.y + ty;
    __shared__ float shared_vec[16];
    if (tx==0 && yIndex<height)
        shared_vec[ty] = norm[yIndex];
    __syncthreads();
    if (xIndex<width && yIndex<height)
        array[yIndex*pitch+xIndex] += shared_vec[ty];
}


/**
 * Adds the query points norm (row vector) to the k first lines of the input
 * array and computes the square root of the resulting values.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param k       number of neighbors to consider
 * @param norm     query points norm stored as a row vector
 */
__global__ void add_query_points_norm_and_sqrt(float * array, int width, int pitch, int k, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        array[yIndex*pitch + xIndex] = sqrt(array[yIndex*pitch + xIndex] + norm[xIndex]);
}

template <typename T>
__global__ void forward(CudaLaunchConfig cfg, T* __restrict__ Z, const int N,
                        const T* __restrict__ X, const T* __restrict__ Y,
                        const T bias) {
  // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x
  // * gridDim.x) {
  for (int i : CudaGridRangeX(cfg.virtual_thread_count)) {
    Z[i] = X[i] + Y[i] + (T)bias;
  }
}

__global__ void compute_distances2(
        const float* __restrict__ ref, const int ref_width,
        const float* __restrict__ query, const int query_width,
        int height,
        float* __restrict__ dist) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    const int ref_pitch = ref_width;
    const int query_pitch = query_width;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height-1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
    int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array
    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
    }

}

//__global__ void compute_distances3(
//        Cuda2DLaunchConfig cfg,
//        const float* __restrict__ ref, const int ref_width,
//        const float* __restrict__ query, const int query_width,
//        int height,
//        float* __restrict__ dist) {
//
//    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
//    __shared__ float shared_A[cfg.thread_per_block.y][cfg.thread_per_block.x];
//    __shared__ float shared_B[cfg.thread_per_block.y][cfg.thread_per_block.x];
//
//    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
//    __shared__ int begin_A;
//    __shared__ int begin_B;
//    __shared__ int step_A;
//    __shared__ int step_B;
//    __shared__ int end_A;
//
//    const int ref_pitch = ref_width;
//    const int query_pitch = query_width;
//
//    // Thread index
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//    // Initializarion of the SSD for the current thread
//    float ssd = 0.f;
//
//    // Loop parameters
//    begin_A = cfg.thread_per_block.y * blockIdx.y;
//    begin_B = cfg.thread_per_block.x * blockIdx.x;
//    step_A  = cfg.thread_per_block.y * ref_pitch;
//    step_B  = cfg.thread_per_block.x * query_pitch;
//    end_A   = begin_A + (height-1) * ref_pitch;
//
//    // Conditions
//    int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
//    int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array
//    int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix
//
//    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
//    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
//
//        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
//        if (a/ref_pitch + ty < height) {
//            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
//            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
//        }
//        else {
//            shared_A[ty][tx] = 0;
//            shared_B[ty][tx] = 0;
//        }
//
//        // Synchronize to make sure the matrices are loaded
//        __syncthreads();
//
//        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
//        if (cond2 && cond1) {
//            for (int k = 0; k < cfg.thread_per_block.y; ++k){
//                float tmp = shared_A[k][ty] - shared_B[k][tx];
//                ssd += tmp*tmp;
//            }
//        }
//
//        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
//        __syncthreads();
//    }
//
//    // Write the block sub-matrix to device memory; each thread writes one element
//    if (cond2 && cond1) {
//        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
//    }
//
//}

template <typename Dtype>
struct KNNFunctor<GPUDevice, Dtype> {
  static void launch(::tensorflow::OpKernelContext* context,
                     const Tensor& t_ref,
                     const Tensor& t_query,
                     Tensor* distances,
                     Tensor* indices,
                     std::int32_t k) {


    const int N = t_ref.NumElements();
    const GPUDevice& d = context->eigen_gpu_device();

//    const auto tensor_ref = t_ref.tensor<Dtype, 2>();
//    const auto tensor_query = t_ref.tensor<Dtype, 2>();

//    ::tensorflow::Cuda2DLaunchConfig cfg =
//        ::tensorflow::GetCuda2DLaunchConfig(static_cast<int>(t_query.dim_size(0)),
//                                            static_cast<int>(t_ref.dim_size(0)),
//                                            d);
//
//    compute_distances2<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
//        cfg,
//        t_ref.flat<Dtype>().data(), t_ref.NumElements(),
//        t_query.flat<Dtype>().data(), t_query.NumElements(),
//        t_ref.dim_size(1),
//        distances->flat<Dtype>().data()
//        );

        const int ref_nb = t_ref.dim_size(1);
        const int query_nb = t_query.dim_size(1);

     // Compute the squared Euclidean distances
     dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
     dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
     if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
     if (ref_nb % BLOCK_DIM != 0) grid0.y += 1;

    compute_distances2<<<grid0, block0, 0, d.stream()>>>(
        t_ref.flat<Dtype>().data(), ref_nb,
        t_query.flat<Dtype>().data(), t_query.dim_size(1),
        t_ref.dim_size(0),
        distances->flat<Dtype>().data()
        );

    // Sort the distances with their respective indexes
    dim3 block1(256, 1, 1);
    dim3 grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0) grid1.x += 1;
    modified_insertion_sort<<<grid1, block1, 0, d.stream()>>>(
            distances->flat<Dtype>().data(), query_nb,
            indices->flat<std::int32_t>().data(), query_nb,
            query_nb, ref_nb, k);

    // Compute the square root of the k smallest distances
    dim3 block2(16, 16, 1);
    dim3 grid2(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0) grid2.x += 1;
    if (k % 16 != 0)        grid2.y += 1;
    compute_sqrt<<<grid2, block2>>>(distances->flat<Dtype>().data(), query_nb, query_nb, k);

    if (!d.ok()) {
      context->SetStatus(tensorflow::errors::Internal(
          "Failed launching MatrixAddGrad on GPU"));
    }

//    // Compute the squared Euclidean distances
//    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
//      dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
//    if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
//    if (ref_nb   % BLOCK_DIM != 0) grid0.y += 1;
//    compute_distances<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev);
//    if (cudaGetLastError() != cudaSuccess) {
//        printf("ERROR: Unable to execute kernel\n");
//        cudaFree(ref_dev);
//        cudaFree(query_dev);
//        cudaFree(dist_dev);
//        cudaFree(index_dev);
//        return false;
//    }
//
//    // Sort the distances with their respective indexes
//    dim3 block1(256, 1, 1);
//    dim3 grid1(query_nb / 256, 1, 1);
//    if (query_nb % 256 != 0) grid1.x += 1;
//    modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
//    if (cudaGetLastError() != cudaSuccess) {
//        printf("ERROR: Unable to execute kernel\n");
//        cudaFree(ref_dev);
//        cudaFree(query_dev);
//        cudaFree(dist_dev);
//        cudaFree(index_dev);
//        return false;
//    }
//
//    // Compute the square root of the k smallest distances
//    dim3 block2(16, 16, 1);
//    dim3 grid2(query_nb / 16, k / 16, 1);
//    if (query_nb % 16 != 0) grid2.x += 1;
//    if (k % 16 != 0)        grid2.y += 1;
//    compute_sqrt<<<grid2, block2>>>(dist_dev, query_nb, query_pitch, k);
//    if (cudaGetLastError() != cudaSuccess) {
//        printf("ERROR: Unable to execute kernel\n");
//        cudaFree(ref_dev);
//        cudaFree(query_dev);
//        cudaFree(dist_dev);
//        cudaFree(index_dev);
//        return false;
//    }
  }
};

template struct KNNFunctor<GPUDevice, float>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
