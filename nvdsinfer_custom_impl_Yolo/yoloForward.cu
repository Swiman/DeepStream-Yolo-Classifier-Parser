/*
 * Created by amin malekmo
 * https://www.github.com/swiman
 */

#include <stdint.h>

inline __device__ float sigmoidGPU(const float &x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayer(const float *input, float *scores, const uint outputSize, const uint batchSize)
{
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint batch_id = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_id >= outputSize || batch_id >= batchSize)
    return;
  scores[batch_id * outputSize + x_id] = sigmoidGPU(input[x_id]);
}

cudaError_t cudaYoloLayer(const void *input, void *scores, const uint &batchSize,
                          const uint64_t &outputSize, cudaStream_t stream);

cudaError_t cudaYoloLayer(const void *input, void *scores, const uint &batchSize,
                          const uint64_t &outputSize, cudaStream_t stream)
{

  int numBlocks = 1;
  dim3 threads_per_block(batchSize, outputSize);

  
  gpuYoloLayer<<<numBlocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const float *>(input),
        reinterpret_cast<float *>(scores),
        outputSize, batchSize);
  return cudaGetLastError();
}
