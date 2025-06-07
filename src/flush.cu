
#include "../include/utils.cuh"
#include <vector>
__device__ void flush_memory(uint32_t *shared_buffer,
                             uint32_t *shared_buffer_size,
                             uint32_t *next_frontier, uint32_t *next_frontier_size) {
  __syncthreads();
  if (threadIdx.x == 0 && *shared_buffer_size > 0) {
    uint32_t index = atomicAdd(next_frontier_size, *shared_buffer_size);
    for (int i = 0; i < *shared_buffer_size; i++) {
      next_frontier[index + i] = shared_buffer[i];
    }
    *shared_buffer_size = 0;
  }
  __syncthreads();
}

__global__ void bfs_our_kernel_baseline(
    const uint32_t *row_offsets,
    const uint32_t *col_indices,
    int *distances,
    const uint32_t *frontier,
    uint32_t *next_frontier,
    uint32_t frontier_size,
    uint32_t current_level,
    uint32_t *next_frontier_size
) {
  __shared__ uint32_t shared_mem[64];
  __shared__ uint32_t size;
  if(threadIdx.x == 0) size = 0;
  __syncthreads();

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontier_size) return;

  uint32_t node = frontier[tid];
  uint32_t row_start = row_offsets[node];
  uint32_t row_end = row_offsets[node + 1];
  for (uint32_t i = row_start; i < row_end; i++) {
    uint32_t neighbor = col_indices[i];
    if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
      uint32_t idx = atomicAdd(&size, 1);
      if (idx < 64) {
        shared_mem[idx] = neighbor;
      }
      // Only flush when buffer is full
      if (idx == 63) {
        flush_memory(shared_mem, &size, next_frontier, next_frontier_size);
      }
    }
  }
  // Final flush for leftovers
  flush_memory(shared_mem, &size, next_frontier, next_frontier_size);
}

void gpu_our_baseline(const uint32_t N, const uint32_t M,
                      const uint32_t *h_rowptr, const uint32_t *h_colidx,
                      const uint32_t source, int *h_distances, bool symmetric) {
  float tot_time = 0.0;
  CUDA_TIMER_INIT(H2D_copy)

  // Allocate and copy graph to device
  uint32_t *d_row_offsets;
  uint32_t *d_col_indices;
  CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  // Allocate memory for distances and frontier queues
  int *d_distances;
  uint32_t *d_frontier;
  uint32_t *d_next_frontier;
  uint32_t *d_next_frontier_size;
  CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

  std::vector<uint32_t> h_frontier(N);
  h_frontier[0] = source;

  CHECK_CUDA(cudaMemcpy(d_frontier, h_frontier.data(), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  // Initialize all distances to -1 (unvisited), and source distance to 0
  CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
  int zero = 0;
  CHECK_CUDA(cudaMemcpy(d_distances + source, &zero, sizeof(int),
                        cudaMemcpyHostToDevice)); // set to 0

  CUDA_TIMER_STOP(H2D_copy)
  tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
  CUDA_TIMER_DESTROY(H2D_copy)

  uint32_t current_frontier_size = 1;
  uint32_t level = 0;

  // Main BFS loop
  CPU_TIMER_INIT(BASELINE_BFS)
  while (current_frontier_size > 0) {

    // Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

    uint32_t block_size = 32;
    uint32_t num_blocks = CEILING(current_frontier_size, block_size);

    // CUDA_TIMER_INIT(BFS_kernel)
    bfs_our_kernel_baseline<<<num_blocks, block_size>>>(
        d_row_offsets, d_col_indices, d_distances, d_frontier, d_next_frontier,
        current_frontier_size, level, d_next_frontier_size);
    CHECK_CUDA(cudaGetLastError());
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
    printf("level %u\n", level);
  }
  CPU_TIMER_STOP(BASELINE_BFS)
  tot_time += CPU_TIMER_ELAPSED(BASELINE_BFS);

  CUDA_TIMER_INIT(D2H_copy)
  CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_TIMER_STOP(D2H_copy)
  tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
  CUDA_TIMER_DESTROY(D2H_copy)

  printf("\n[OUT] Total BFS time: %f ms\n", tot_time);
  printf("[OUT] Graph diameter: %u\n", level);

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
}
