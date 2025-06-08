#include "../include/utils.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;
// Kernel: Process each node in the frontier and add unvisited neighbors to
// next_frontier
#define FRONTIER_SIZE 24
#define MAX_LEVELS 100000
__global__ void bfs_kernel_rec(const uint32_t *row_offsets,
                               const uint32_t *col_indices, int *distances,
                               uint32_t *next_frontier,
                               uint32_t next_frontier_size,
                               uint32_t *next_levels, uint32_t *counter) {
  uint32_t frontier[FRONTIER_SIZE];
  uint32_t levels[FRONTIER_SIZE];
  uint32_t frontier_size;
  // Copy next_frontier to frontier in parallel
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    frontier_size = next_frontier_size;
    next_frontier_size = 0;
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    for (uint32_t i = threadIdx.x; i < frontier_size; i += blockDim.x) {
      frontier[i] = next_frontier[i];
      levels[i] = distances[frontier[i]];
    }
  }
  __syncthreads();

  if (threadIdx.x >= frontier_size)
    return;

  uint32_t node = frontier[threadIdx.x];
  uint32_t row_start = row_offsets[node];
  uint32_t row_end = row_offsets[node + 1];
  uint32_t level = levels[node];
  uint32_t written_out = 0;

  for (uint32_t i = row_start + threadIdx.y; i < row_end; i += blockDim.y) {
    uint32_t neighbor = col_indices[i];
    if (atomicMin(&distances[neighbor], level + 1) > level + 1) {
      // If the neighbor was unvisited, add it to the next frontier
      uint32_t index = atomicAdd(&next_frontier_size, 1);
      if (index < FRONTIER_SIZE) {
        next_frontier[index] = neighbor;
        next_levels[index] = level + 1;
        written_out++;
      }
    }
  }
}

void gpu_bfs_coalesced_rec(const uint32_t N, const uint32_t M,
                           const uint32_t *h_rowptr, const uint32_t *h_colidx,
                           const uint32_t source, int *h_distances) {
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0);
  cudaDeviceSetMemPool(0, pool);
  // Warm up the async allocator (do this before your timed section)
  void *dummy_ptr = nullptr;
  cudaStream_t dummy_stream;
  cudaStreamCreate(&dummy_stream);
  cudaMallocAsync(&dummy_ptr, 1, dummy_stream); // 1 byte is enough
  cudaFreeAsync(dummy_ptr, dummy_stream);
  cudaStreamSynchronize(dummy_stream);
  cudaStreamDestroy(dummy_stream);

  float tot_time = 0.0;
  CUDA_TIMER_INIT(H2D_copy)
  cudaStream_t stream_row, stream_col, stream_frontier, stream_next_frontier,
      stream_distances, stream_while;
  CHECK_CUDA(cudaStreamCreate(&stream_row));
  CHECK_CUDA(cudaStreamCreate(&stream_col));
  CHECK_CUDA(cudaStreamCreate(&stream_frontier));
  CHECK_CUDA(cudaStreamCreate(&stream_next_frontier));
  CHECK_CUDA(cudaStreamCreate(&stream_distances));
  CHECK_CUDA(cudaStreamCreate(&stream_while));

  // Allocate and copy graph to device
  uint32_t *d_row_offsets;
  uint32_t *d_col_indices;
  /*CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));*/
  CHECK_CUDA(
      cudaMallocAsync(&d_row_offsets, (N + 1) * sizeof(uint32_t), stream_row));
  CHECK_CUDA(cudaMemcpyAsync(d_row_offsets, h_rowptr,
                             (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice,
                             stream_row));
  CHECK_CUDA(cudaMallocAsync(&d_col_indices, M * sizeof(uint32_t), stream_col));

  CHECK_CUDA(cudaMemcpyAsync(d_col_indices, h_colidx, M * sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_col));

  // Allocate memory for distances and frontier queues
  int *d_distances;
  uint32_t *d_frontier;
  uint32_t *d_next_frontier;
  uint32_t *d_next_frontier_size;
  /*CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));*/
  CHECK_CUDA(cudaMallocAsync(&d_distances, N * sizeof(int), stream_distances));
  CHECK_CUDA(
      cudaMallocAsync(&d_frontier, N * sizeof(uint32_t), stream_frontier));
  CHECK_CUDA(cudaMallocAsync(&d_next_frontier, N * sizeof(uint32_t),
                             stream_next_frontier));
  CHECK_CUDA(cudaMallocAsync(&d_next_frontier_size, sizeof(uint32_t),
                             stream_next_frontier));

  /*std::vector<uint32_t> h_frontier(1);
  h_frontier[0] = source;*/

  /*CHECK_CUDA(cudaMemcpy(d_frontier, &source, sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  // Initialize all distances to -1 (unvisited), and source distance to 0
  CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
  CHECK_CUDA(cudaMemset(d_distances + source, 0, sizeof(int))); // set to 0*/
  CHECK_CUDA(cudaMemcpyAsync(d_frontier, &source, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_frontier));
  CHECK_CUDA(
      cudaMemsetAsync(d_distances, -1, N * sizeof(int), stream_distances));
  CHECK_CUDA(
      cudaMemsetAsync(d_distances + source, 0, sizeof(int), stream_distances));

  CHECK_CUDA(cudaStreamSynchronize(stream_frontier));
  CHECK_CUDA(cudaStreamSynchronize(stream_distances));
  CHECK_CUDA(cudaStreamSynchronize(stream_next_frontier));

  CHECK_CUDA(cudaStreamSynchronize(stream_row));
  CHECK_CUDA(cudaStreamSynchronize(stream_col));

  CUDA_TIMER_STOP(H2D_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(H2D_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
  printf("after init %f\n", tot_time);
  CUDA_TIMER_DESTROY(H2D_copy)

  uint32_t current_frontier_size = 1;
  uint32_t level = 0;

  // Main BFS loop
  CPU_TIMER_INIT(BASELINE_BFS)
  while (current_frontier_size > 0) {

#ifdef DEBUG_PRINTS
    printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n",
           is_placeholder ? "" : " BASELINE", level, current_frontier_size);
#endif
#ifdef ENABLE_NVTX
    // Mark start of level in NVTX
    nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
#endif

    // Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

    // CUDA_TIMER_INIT(BFS_kernel)
    dim3 block_dim(24, 32);
    dim3 grid_dim(CEILING(current_frontier_size, block_dim.x));
    bfs_kernel_coalesced_shared_copy<<<grid_dim, block_dim>>>(
        d_row_offsets, d_col_indices, d_distances, d_frontier, d_next_frontier,
        current_frontier_size, level, d_next_frontier_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    // CUDA_TIMER_STOP(BFS_kernel)
    // #ifdef DEBUG_PRINTS
    //   CUDA_TIMER_PRINT(BFS_kernel)
    // #endif
    // CUDA_TIMER_DESTROY(BFS_kernel)

    // Swap frontier pointers
    std::swap(d_frontier, d_next_frontier);

    // Copy size of next frontier to host
    CHECK_CUDA(cudaMemcpy(&current_frontier_size, d_next_frontier_size,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    level++;
    ;

#ifdef ENABLE_NVTX
    // End NVTX range for level
    nvtxRangePop();
#endif
  }
  CPU_TIMER_STOP(BASELINE_BFS)
#ifdef DEBUG_PRINTS
  CPU_TIMER_PRINT(BASELINE_BFS)
#endif
  tot_time += CPU_TIMER_ELAPSED(BASELINE_BFS);
  printf("after coalesced %f\n", tot_time);
  CUDA_TIMER_INIT(D2H_copy)
  CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_TIMER_STOP(D2H_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(D2H_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
  CUDA_TIMER_DESTROY(D2H_copy)

  printf("\n[OUT] Total BFS time: %f ms\n" RESET, tot_time);

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
}