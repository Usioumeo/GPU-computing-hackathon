#include "../include/utils.cuh"
#include <sys/types.h>
// Kernel: Process each node in the frontier and add unvisited neighbors to
// next_frontier
#define SHARED_BUFFER_SIZE 10000 // Adjusted for shared memory size
__global__ void bfs_kernel_coalesced_shared_faster(
    const uint32_t *__restrict__ row_offsets,
    const uint32_t *__restrict__ col_indices, int *__restrict__ distances,
    const uint32_t *__restrict__ frontier, uint32_t *__restrict__ next_frontier,
    uint32_t *frontier_size, uint32_t current_level,
    uint32_t *__restrict__ next_frontier_size) {
  __shared__ uint32_t shared_buffer[SHARED_BUFFER_SIZE];
  __shared__ uint32_t shared_buffer_size;
  __shared__ uint32_t start_index;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    shared_buffer_size =
        0; // Reset shared buffer size at the start of each block
  }
  __syncthreads();
  while (tid < *frontier_size) {

    uint32_t node = frontier[tid];
    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];

    for (uint32_t i = row_start + threadIdx.y; i < row_end; i += blockDim.y) {
      uint32_t neighbor = col_indices[i];
      if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
        uint32_t index_shared = atomicAdd(&shared_buffer_size, 1);
        if (index_shared < SHARED_BUFFER_SIZE) {
          shared_buffer[index_shared] = neighbor;
        } else {
          // If shared buffer is full, flush it to next_frontier
          uint32_t index = atomicAdd(next_frontier_size, 1);
          next_frontier[index] = neighbor;
        }
      }
    }
    __syncthreads();
    uint32_t flush_size = min(shared_buffer_size, SHARED_BUFFER_SIZE);
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      start_index = atomicAdd(next_frontier_size, flush_size);
    }
    __syncthreads();

    for (uint32_t i = threadIdx.y + blockDim.y * threadIdx.x; i < flush_size;
         i += blockDim.y) {
      next_frontier[start_index + i] = shared_buffer[i];
    }
    __syncthreads();

    // Reset the shared buffer for the next batch
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      shared_buffer_size = 0;
    }
    __syncthreads();

    tid += blockDim.x * gridDim.x;
  }
}

void gpu_bfs_coalesced_shared_faster(const uint32_t N, const uint32_t M,
                                     const uint32_t *h_rowptr,
                                     const uint32_t *h_colidx,
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
      stream_distances;
  CHECK_CUDA(cudaStreamCreate(&stream_row));
  CHECK_CUDA(cudaStreamCreate(&stream_col));
  CHECK_CUDA(cudaStreamCreate(&stream_frontier));
  CHECK_CUDA(cudaStreamCreate(&stream_next_frontier));
  CHECK_CUDA(cudaStreamCreate(&stream_distances));

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
  uint32_t *d_frontier_size;
  uint32_t *d_next_frontier;
  uint32_t *d_next_frontier_size;
  /*CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));*/
  CHECK_CUDA(cudaMallocAsync(&d_distances, N * sizeof(int), stream_distances));
  CHECK_CUDA(
      cudaMallocAsync(&d_frontier, N * sizeof(uint32_t), stream_frontier));
  CHECK_CUDA(
      cudaMallocAsync(&d_frontier_size, sizeof(uint32_t), stream_frontier));
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
  uint32_t dummy = 1;
  CHECK_CUDA(cudaMemcpyAsync(d_frontier_size, &dummy, sizeof(uint32_t),
                             cudaMemcpyHostToDevice, stream_frontier));
  CHECK_CUDA(cudaMemsetAsync(d_next_frontier_size, 0, sizeof(uint32_t),
                             stream_next_frontier));
  CHECK_CUDA(
      cudaMemsetAsync(d_distances, -1, N * sizeof(int), stream_distances));
  CHECK_CUDA(
      cudaMemsetAsync(d_distances + source, 0, sizeof(int), stream_distances));

  CHECK_CUDA(cudaStreamSynchronize(stream_frontier));
  CHECK_CUDA(cudaStreamSynchronize(stream_distances));
  CHECK_CUDA(cudaStreamSynchronize(stream_next_frontier));

  CHECK_CUDA(cudaStreamSynchronize(stream_row));
  CHECK_CUDA(cudaStreamSynchronize(stream_col));

  cudaStream_t stream_kernel, stream_memcpy;
  cudaEvent_t kernel_done;
  cudaStreamCreate(&stream_kernel);
  cudaStreamCreate(&stream_memcpy);
  cudaEventCreate(&kernel_done);

  CUDA_TIMER_STOP(H2D_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(H2D_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
  printf("after init %f\n", tot_time);
  CUDA_TIMER_DESTROY(H2D_copy)

  uint32_t level = 0;
  uint32_t host_frontier_size = 1;
  // Main BFS loop
  CPU_TIMER_INIT(BASELINE_BFS)
  while (host_frontier_size > 0) {

#ifdef DEBUG_PRINTS
    printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n",
           is_placeholder ? "" : " BASELINE", level, current_frontier_size);
#endif
#ifdef ENABLE_NVTX
    // Mark start of level in NVTX
    nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
#endif
    printf("level %u\n", level);

    /*// Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

    // CUDA_TIMER_INIT(BFS_kernel)
    dim3 block_dim(24, 32);
    dim3 grid_dim(568);
    bfs_kernel_coalesced_shared_faster<<<grid_dim, block_dim, 0,
                                         stream_kernel>>>(
        d_row_offsets, d_col_indices, d_distances, d_frontier, d_next_frontier,
        d_frontier_size, level++, d_next_frontier_size);

    cudaMemsetAsync(d_frontier_size, 0, sizeof(uint32_t), stream_memcpy);
    // Record event when kernel is done
    cudaEventRecord(kernel_done, stream_kernel);

    // Make memcpy stream wait for kernel to finish
    cudaStreamWaitEvent(stream_memcpy, kernel_done, 0);

    bfs_kernel_coalesced_shared_faster<<<grid_dim, block_dim, 0,
                                         stream_kernel>>>(
        d_row_offsets, d_col_indices, d_distances, d_next_frontier, d_frontier,
        d_next_frontier_size, level++, d_frontier_size);

    // Async copy on stream_memcpy (will start only after kernel_done)
    cudaMemcpyAsync(&host_frontier_size, d_next_frontier_size, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream_memcpy);
    // Wait for memcpy to finish before host decision
    cudaStreamSynchronize(stream_memcpy);

#ifdef ENABLE_NVTX
    // End NVTX range for level
    nvtxRangePop();
#endif*/
    // Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(&host_frontier_size, d_frontier_size,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // CUDA_TIMER_INIT(BFS_kernel)
    dim3 block_dim(24, 32);
    dim3 grid_dim(576);
    bfs_kernel_coalesced_shared_faster<<<grid_dim, block_dim>>>(
        d_row_offsets, d_col_indices, d_distances, d_frontier, d_next_frontier,
        d_frontier_size, level, d_next_frontier_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    // CUDA_TIMER_STOP(BFS_kernel)
    // #ifdef DEBUG_PRINTS
    //   CUDA_TIMER_PRINT(BFS_kernel)
    // #endif
    // CUDA_TIMER_DESTROY(BFS_kernel)

    // exit(0);
    //  Copy size of next frontier to host
    CHECK_CUDA(cudaMemcpy(&host_frontier_size, d_next_frontier_size,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // Swap frontier pointers
    std::swap(d_frontier, d_next_frontier);
    std::swap(d_frontier_size, d_next_frontier_size);
    level++;
    ;
  }
  cudaDeviceSynchronize();
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