#include <vector>
#include <queue>
#include <string>
#include "utils.cuh"

// Kernel: Process each node in the frontier and add unvisited neighbors to next_frontier
__global__ void bfs_kernel_baseline(
  const uint32_t* row_offsets,       // CSR row offsets
  const uint32_t* col_indices,       // CSR column indices (neighbors)
  int* distances,                    // Output distances array
  const uint32_t* frontier,          // Current frontier
  uint32_t* next_frontier,           // Next frontier to populate
  uint32_t frontier_size,            // Size of current frontier
  uint32_t current_level,            // BFS level (depth)
  uint32_t* next_frontier_size       // Counter for next frontier
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontier_size) return;

  uint32_t node = frontier[tid];
  uint32_t row_start = row_offsets[node];
  uint32_t row_end = row_offsets[node + 1];

  for (uint32_t i = row_start; i < row_end; i++) {
    uint32_t neighbor = col_indices[i];

    // Use atomic compare-and-swap to avoid revisiting nodes
    if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
      // Atomically add the neighbor to the next frontier
      uint32_t index = atomicAdd(next_frontier_size, 1);
      next_frontier[index] = neighbor;
    }
  }
}

void gpu_bfs_baseline(
  const uint32_t N,
  const uint32_t M,
  const uint32_t *h_rowptr,
  const uint32_t *h_colidx,
  const uint32_t source,
  int *h_distances,
  bool is_placeholder
) {
  float tot_time = 0.0;
  CUDA_TIMER_INIT(H2D_copy)

  // Allocate and copy graph to device
  uint32_t* d_row_offsets; uint32_t* d_col_indices;
  CHECK_CUDA(cudaMalloc(&d_row_offsets, (N + 1) * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, M * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(d_row_offsets, h_rowptr, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, h_colidx, M * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Allocate memory for distances and frontier queues
  int* d_distances; uint32_t* d_frontier; uint32_t* d_next_frontier; uint32_t* d_next_frontier_size;
  CHECK_CUDA(cudaMalloc(&d_distances, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_next_frontier_size, sizeof(uint32_t)));

  std::vector<uint32_t> h_frontier(N);
  h_frontier[0] = source;

  CHECK_CUDA(cudaMemcpy(d_frontier, h_frontier.data(), sizeof(uint32_t), cudaMemcpyHostToDevice));
  // Initialize all distances to -1 (unvisited), and source distance to 0
  CHECK_CUDA(cudaMemset(d_distances, -1, N * sizeof(int)));
  CHECK_CUDA(cudaMemset(d_distances + source, 0, sizeof(int))); // set to 0

  CUDA_TIMER_STOP(H2D_copy)
  #ifdef DEBUG_PRINTS
    CUDA_TIMER_PRINT(H2D_copy)
  #endif
  tot_time += CUDA_TIMER_ELAPSED(H2D_copy);
  CUDA_TIMER_DESTROY(H2D_copy)

  uint32_t current_frontier_size = 1;
  uint32_t level = 0;

  // Main BFS loop
  CPU_TIMER_INIT(BASELINE_BFS)
  while (current_frontier_size > 0) {

    #ifdef DEBUG_PRINTS
      printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n", is_placeholder ? "" : " BASELINE", level, current_frontier_size);
    #endif
    #ifdef ENABLE_NVTX
      // Mark start of level in NVTX
      nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
    #endif

    // Reset counter for next frontier
    CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

    uint32_t block_size = 512;
    uint32_t num_blocks = CEILING(current_frontier_size, block_size);

    // CUDA_TIMER_INIT(BFS_kernel)
    bfs_kernel_baseline<<<num_blocks, block_size>>>(
      d_row_offsets,
      d_col_indices,
      d_distances,
      d_frontier,
      d_next_frontier,
      current_frontier_size,
      level,
      d_next_frontier_size
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    // CUDA_TIMER_STOP(BFS_kernel)
    // #ifdef DEBUG_PRINTS
    //   CUDA_TIMER_PRINT(BFS_kernel)
    // #endif
    // CUDA_TIMER_DESTROY(BFS_kernel)

    // Swap frontier pointers
    std::swap(d_frontier, d_next_frontier);

    // Copy size of next frontier to host
    CHECK_CUDA(cudaMemcpy(&current_frontier_size, d_next_frontier_size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    level++;

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

  CUDA_TIMER_INIT(D2H_copy)
  CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_TIMER_STOP(D2H_copy)
  #ifdef DEBUG_PRINTS
    CUDA_TIMER_PRINT(D2H_copy)
  #endif
  tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
  CUDA_TIMER_DESTROY(D2H_copy)

  printf("\n[OUT] Total%s BFS time: %f ms\n", is_placeholder ? "" : " BASELINE", tot_time);
  if (!is_placeholder) printf("[OUT] Graph diameter: %u\n", level);

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
}

void cpu_bfs_baseline(
  const uint32_t N,
  const uint32_t *h_rowptr,
  const uint32_t *h_colidx,
  const uint32_t source,
  int *h_distances
) {
  std::vector<bool> visited(N, false);
  std::queue<uint32_t> q;

  memset(h_distances, -1, N * sizeof(int));

  h_distances[source] = 0;
  visited[source] = true;
  q.push(source);

  while (!q.empty()) {
    uint32_t current = q.front();
    q.pop();

    for (uint32_t i = h_rowptr[current]; i < h_rowptr[current + 1]; ++i) {
      uint32_t neighbor = h_colidx[i];
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        h_distances[neighbor] = h_distances[current] + 1;
        q.push(neighbor);
      }
    }
  }
}