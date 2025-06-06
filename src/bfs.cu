// #define ENABLE_NVTX
// #define ENABLE_CPU_BASELINE
// #define DEBUG_PRINTS
#include <sys/types.h>
#define ENABLE_CORRECTNESS_CHECK

#define EXIT_INCORRECT_DISTANCES 10

#include <cuda_runtime.h>
#include <stdio.h>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "../distributed_mmio/include/mmio.h"
#include "../distributed_mmio/include/mmio_utils.h"

#include "../include/bfs_baseline.cuh"
#include "../include/cli.hpp"
#include "../include/colors.h"
#include "../include/mt19937-64.hpp"
#include "../include/utils.cuh"

// Kernel: Process each node in the frontier and add unvisited neighbors to
// next_frontier
__global__ void bfs_our_kernel_baseline(
    const uint32_t *row_offsets, // CSR row offsets
    const uint32_t *col_indices, // CSR column indices (neighbors)
    int *distances,              // Output distances array
    const uint32_t *frontier,    // Current frontier
    uint32_t *next_frontier,     // Next frontier to populate
    uint32_t frontier_size,      // Size of current frontier
    uint32_t current_level,      // BFS level (depth)
    uint32_t *next_frontier_size // Counter for next frontier
) {
  uint32_t shared_mem[32];
  uint32_t size=0;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= frontier_size)
    return;

  uint32_t node = frontier[tid];
  uint32_t row_start = row_offsets[node];
  uint32_t row_end = row_offsets[node + 1];
  for (uint32_t i = row_start; i < row_end; i++) {
    uint32_t neighbor = col_indices[i];

    // Use atomic compare-and-swap to avoid revisiting nodes
    if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
      shared_mem[size++] = neighbor;
      if (size ==32) {
        // If shared memory is full, write to next frontier
        uint32_t index = atomicAdd(next_frontier_size, size);
        for (uint32_t j = 0; j < size; j++) {
          next_frontier[index + j] = shared_mem[j];
        }
        size = 0; // Reset size for next batch
      }
      // Atomically add the neighbor to the next frontier
      //uint32_t index = atomicAdd(next_frontier_size, 1);
      //next_frontier[index] = neighbor;
    }
  }
  // If there are remaining neighbors in shared memory, write them to next frontier
  if (size > 0) {
    uint32_t index = atomicAdd(next_frontier_size, size);
    for (uint32_t j = 0; j < size; j++) {
      next_frontier[index + j] = shared_mem[j];
    }
  }
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

    uint32_t block_size = 512;
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
    //printf("level %u\n", level);
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

int main(int argc, char **argv) {
  int return_code = EXIT_SUCCESS;

  Cli_Args args;
  init_cli();
  if (parse_args(argc, argv, &args) != 0) {
    return -1;
  }

  int device_count;
  cudaGetDeviceCount(&device_count);
  if (device_count <= 0) {
    fprintf(stderr, "No GPU available: device_count=%d\n", device_count);
    return EXIT_FAILURE;
  }
  cudaSetDevice(0);
  Matrix_Metadata
      metadata; //=(Matrix_Metadata*)malloc(sizeof(Matrix_Metadata));
  CPU_TIMER_INIT(MTX_read)
  CSR_local<uint32_t, float> *csr = Distr_MMIO_CSR_local_read<uint32_t, float>(
      args.filename, false, &metadata);
  if (csr == NULL) {
    printf("Failed to import graph from file [%s]\n", args.filename);
    return -1;
  }
  CPU_TIMER_STOP(MTX_read)
  printf("\n[OUT] MTX file read time: %f ms\n", CPU_TIMER_ELAPSED(MTX_read));
  printf("Graph size: %.3fM vertices, %.3fM edges\n", csr->nrows / 1e6,
         csr->nnz / 1e6);

  GraphCSR graph;
  graph.row_ptr = csr->row_ptr;
  graph.col_idx = csr->col_idx;
  graph.num_vertices = csr->nrows;
  graph.num_edges = csr->nnz;
  // print_graph_csr(graph);

  uint32_t *sources =
      generate_sources(&graph, args.runs, graph.num_vertices, args.source);
  int *distances_gpu_baseline = (int *)malloc(graph.num_vertices * sizeof(int));
  int *distances = (int *)malloc(graph.num_vertices * sizeof(int));
  bool correct = true;

  for (int source_i = 0; source_i < args.runs; source_i++) {
    uint32_t source = sources[source_i];
    printf("\n[OUT] -- BFS iteration #%u, source=%u --\n", source_i, source);

    // Run the BFS baseline
    gpu_bfs_baseline(graph.num_vertices, graph.num_edges, graph.row_ptr,
                     graph.col_idx, source, distances_gpu_baseline, false);

#ifdef ENABLE_NVTX
    nvtxRangePushA("Complete BFS");
#endif
    gpu_our_baseline(graph.num_vertices, graph.num_edges, graph.row_ptr,
                     graph.col_idx, source, distances, metadata.is_symmetric);
#ifdef ENABLE_NVTX
    nvtxRangePop();
#endif

    bool match = true;
#ifdef ENABLE_CORRECTNESS_CHECK
    for (uint32_t i = 0; i < graph.num_vertices; ++i) {
      if (distances_gpu_baseline[i] != distances[i]) {
        printf(
            "Mismatch at node %u: Baseline distance = %d, Your distance = %d\n",
            i, distances_gpu_baseline[i], distances[i]);
        match = false;
        break;
      }
    }
    if (match) {
      printf(BRIGHT_GREEN "Correctness OK\n" RESET);
    } else {
      printf(BRIGHT_RED
             "GPU and CPU BFS results do not match for source node %u.\n" RESET,
             source);
      return_code = EXIT_INCORRECT_DISTANCES;
      correct = false;
    }
#endif

#ifdef ENABLE_CPU_BASELINE
    int cpu_distances[graph.num_vertices];

    CPU_TIMER_INIT(CPU_BFS)
    cpu_bfs_baseline(graph.num_vertices, graph.row_ptr, graph.col_idx, source,
                     cpu_distances);
    CPU_TIMER_CLOSE(CPU_BFS)

    match = true;
    for (uint32_t i = 0; i < graph.num_vertices; ++i) {
      if (distances_gpu_baseline[i] != cpu_distances[i]) {
        printf("Mismatch at node %u: GPU distance = %d, CPU distance = %d\n", i,
               distances_gpu_baseline[i], cpu_distances[i]);
        match = false;
        break;
      }
    }
    if (match) {
      printf(BRIGHT_GREEN "[CPU] Correctness OK\n" RESET);
    } else {
      printf(BRIGHT_RED
             "GPU and CPU BFS results do not match for source node %u.\n" RESET,
             source);
      return_code = EXIT_INCORRECT_DISTANCES;
    }
#endif
  }

  if (correct)
    printf("\n[OUT] ALL RESULTS ARE CORRECT\n");
  else
    printf(BRIGHT_RED "\nSOME RESULTS ARE WRONG\n" RESET);

  Distr_MMIO_CSR_local_destroy(&csr);
  free(sources);
  free(distances_gpu_baseline);
  free(distances);

  return return_code;
}
