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
/*
float bfs(Vector<float>* v, const Matrix<float>* A, Index s, Descriptor* desc) {
  Index A_nrows;
  CHECK(A->nrows(&A_nrows));
  CHECK(v->fill(0.f));
  Vector<float> f1(A_nrows);
  Vector<float> f2(A_nrows);
  Desc_value desc_value;
  CHECK(desc->get(GrB_MXVMODE, &desc_value));
  if (desc_value == GrB_PULLONLY) {
    CHECK(f1.fill(0.f));
    CHECK(f1.setElement(1.f, s));
  } else {
    std::vector<Index> indices(1, s);
    std::vector<float> values(1, 1.f);
    CHECK(f1.build(&indices, &values, 1, nullptr));
  }
  float iter;
  float succ = 0.f;
  Index unvisited = A_nrows;
  for (iter = 1; iter <= desc->descriptor_.max_niter_; ++iter) {
    if (desc->descriptor_.debug()) {
      printf("=====BFS Iteration %g=====\n", iter - 1);
      v->print();
      f1.print();
    }
    if (iter > 1) {
      const char* vxm_mode = (desc->descriptor_.lastmxv_ == GrB_PUSHONLY) ?
"push" : "pull"; if (desc->descriptor_.timing_ == 1) printf("%g, %g/%d, %d,
%s\n", iter - 1, succ, A_nrows, unvisited, vxm_mode);
    }
    unvisited -= static_cast<int>(succ);
    // assign: v = iter where f1 is nonzero
    for (Index i = 0; i < A_nrows; ++i) if (f1[i]) (*v)[i] = iter;
    // vxm: f2 = f1 * A (Boolean semiring)
    for (Index i = 0; i < A_nrows; ++i) {
      float val = 0.f;
      for (Index j = 0; j < A_nrows; ++j) val = val || (f1[j] && A->get(j, i));
      f2[i] = val && !(*v)[i];
    }
    f1.swap(&f2);
    // reduce: succ = sum(f1)
    succ = 0.f;
    for (Index i = 0; i < A_nrows; ++i) succ += f1[i];
    if (desc->descriptor_.debug()) printf("succ: %g\n", succ);
    if (succ == 0) break;
  }
  return iter;
}*/

__global__ void bfs_kernel_our_baseline(
    const uint32_t *row_offsets, // CSR row offsets
    const uint32_t *col_indices, // CSR column indices (neighbors)
    int *distances,              // Output distances array
    const uint32_t *frontier,    // Current frontier
    uint32_t *next_frontier,     // Next frontier to populate
    uint32_t frontier_size,      // Size of current frontier
    uint32_t current_level,      // BFS level (depth)
    uint32_t *next_frontier_size // Counter for next frontier
) {
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
      // Atomically add the neighbor to the next frontier
      uint32_t index = atomicAdd(next_frontier_size, 1);
      next_frontier[index] = neighbor;
    }
  }
}

__global__ void bfs_kernel_bottom_up(
  const uint32_t *row_offsets,      // CSR row offsets
  const uint32_t *col_indices,      // CSR column indices (neighbors)
  int *distances,                   // Output distances array
  const uint32_t *not_visited,      // List of not visited nodes
  const uint32_t not_visited_size,  // Size of not_visited list (pass by value)
  uint32_t *next_not_visited,       // Output: nodes still not visited
  uint32_t *next_not_visited_size,  // Counter for next_not_visited
  uint32_t current_level            // BFS level (depth)
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= not_visited_size)
    return;
    
  uint32_t node = not_visited[tid];
  uint32_t row_start = row_offsets[node];
  uint32_t row_end = row_offsets[node + 1];
  
  // Check if any neighbor has current_level distance
  for (uint32_t i = row_start; i < row_end; i++) {
    uint32_t neighbor = col_indices[i];
    if (distances[neighbor] == current_level) {
      // Use atomic operation to safely set distance
      if (atomicCAS(&distances[node], -1, current_level + 1) == -1) {
        return; // Successfully set distance
      }
    }
  }
  
  // If not visited in this round, add to next_not_visited
  uint32_t index = atomicAdd(next_not_visited_size, 1);
  next_not_visited[index] = node;
}

__global__ void from_frontier_to_not_visited(
    int *distances,            // Output distances array
    uint32_t *not_visited,     // Next frontier to populate
    uint32_t *not_visited_size // Counter for next frontier
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (distances[tid] == -1) {
    // Atomically add the node to the not_visited list
    uint32_t index = atomicAdd(not_visited_size, 1);
    not_visited[index] = tid;
  } else {
    printf("porco\n");
  }
}
__global__ void from_not_visited_to_frontier(
    int *distances,          // Output distances array
    uint32_t *frontier,      // Next frontier to populate
    uint32_t *frontier_size, // Counter for next frontier
    uint32_t current_level   // Next frontier to populate
) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (distances[tid] == current_level) {
    // Atomically add the node to the not_visited list
    uint32_t index = atomicAdd(frontier_size, 1);
    frontier[index] = tid;
  }
}

void gpu_our_baseline(const uint32_t N,         // Number of veritices
                      const uint32_t M,         // Number of edges
                      const uint32_t *h_rowptr, // Graph CSR rowptr
                      const uint32_t *h_colidx, // Graph CSR colidx
                      const uint32_t source,    // Source veritex
                      int *h_distances          // Write here your distances
) {
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

  uint32_t *not_visited;
  uint32_t *next_not_visited;
  uint32_t *next_not_visited_size;
  CHECK_CUDA(cudaMalloc(&not_visited, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&next_not_visited, N * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&next_not_visited_size, sizeof(uint32_t)));

  std::vector<uint32_t> h_frontier(N);
  h_frontier[0] = source;

  CHECK_CUDA(cudaMemcpy(d_frontier, h_frontier.data(), sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
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
  uint32_t not_visited_size = 0;
  uint32_t level = 0;

  bool top_down = true;
  // Main BFS loop
  CPU_TIMER_INIT(BASELINE_BFS)
  uint32_t total_visited = 1;
  while (total_visited <N) {

#ifdef DEBUG_PRINTS
    printf("[GPU BFS%s] level=%u, current_frontier_size=%u\n",
           is_placeholder ? "" : " BASELINE", level, current_frontier_size);
#endif
#ifdef ENABLE_NVTX
    // Mark start of level in NVTX
    nvtxRangePushA(("BFS Level " + std::to_string(level)).c_str());
#endif
    if (top_down) {
      // Reset counter for next frontier
      CHECK_CUDA(cudaMemset(d_next_frontier_size, 0, sizeof(uint32_t)));

      uint32_t block_size = 512;
      uint32_t num_blocks = CEILING(current_frontier_size, block_size);

      // CUDA_TIMER_INIT(BFS_kernel)
      bfs_kernel_our_baseline<<<num_blocks, block_size>>>(
          d_row_offsets, d_col_indices, d_distances, d_frontier,
          d_next_frontier, current_frontier_size, level, d_next_frontier_size);
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
      total_visited += current_frontier_size;
      printf("Total visited 1= %d %d\n", total_visited, current_frontier_size);
    } else {
      printf("%d %d %d\n", level, total_visited, not_visited_size);
      // todo remove
      if (level >= 40) {
        exit(0);
      }
      CHECK_CUDA(cudaMemset(next_not_visited_size, 0, sizeof(uint32_t)));
      uint32_t block_size = 512;
      uint32_t num_blocks = CEILING(not_visited_size, block_size);
      // bottom up bfs
      bfs_kernel_bottom_up<<<num_blocks, block_size>>>(
          d_row_offsets, d_col_indices, d_distances, not_visited,
          not_visited_size, next_not_visited, next_not_visited_size, level);

      CHECK_CUDA(cudaDeviceSynchronize());
      // Swap frontier pointers
      std::swap(not_visited, next_not_visited);
      // Copy size of next frontier to host
      CHECK_CUDA(cudaMemcpy(&not_visited_size, next_not_visited_size,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaDeviceSynchronize());
      // TODO maybe it is not N
      printf("not visited size at iteration(%d) %d\n", level, not_visited_size);
      total_visited = N - not_visited_size;
      printf("post tot visit %d\n", total_visited);
    }
    level++;
    // check euristic
    if (top_down) { // total_visited >=M/14
      top_down = false;
      not_visited_size = 0;
      int block_size = 512;
      int num_blocks = CEILING(N, block_size);
      printf("switching euristic\n");
      // Reset next_not_visited_size to 0 before kernel launch
      CHECK_CUDA(cudaMemset(next_not_visited_size, 0, sizeof(uint32_t)));
      from_frontier_to_not_visited<<<num_blocks, block_size>>>(
          d_distances, next_not_visited, next_not_visited_size);

      CHECK_CUDA(cudaDeviceSynchronize());
      std::swap(not_visited, next_not_visited);
      CHECK_CUDA(cudaMemcpy(&not_visited_size, next_not_visited_size,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost));
      // Copy size of next frontier to host
      printf("Total visited = %d %d %d\n", total_visited, N - not_visited_size,
             N);

      // exit(-1);
    }
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
  CHECK_CUDA(cudaMemcpy(h_distances, d_distances, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_TIMER_STOP(D2H_copy)
#ifdef DEBUG_PRINTS
  CUDA_TIMER_PRINT(D2H_copy)
#endif
  tot_time += CUDA_TIMER_ELAPSED(D2H_copy);
  CUDA_TIMER_DESTROY(D2H_copy)

  printf("\n[OUT] Total BFS time: %f ms\n", tot_time);

  // Free device memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_distances);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
}

void gpu_bfs(const uint32_t N,         // Number of veritices
             const uint32_t M,         // Number of edges
             const uint32_t *h_rowptr, // Graph CSR rowptr
             const uint32_t *h_colidx, // Graph CSR colidx
             const uint32_t source,    // Source veritex
             int *h_distances          // Write here your distances
) {
  /***********************
   * IMPLEMENT HERE YOUR CUDA BFS
   * Feel free to structure you code (i.e. create other files, macros etc.)
   * *********************/

  // !! This is just a placeholder !!
  gpu_bfs_baseline(N, M, h_rowptr, h_colidx, source, h_distances, true);

  // !! This is an example of how to keep track of runtime. Make sure to include
  // everything. !!
  /*float tot_time = 0.0f;
  CPU_TIMER_INIT(BFS_preprocess)
  //<<< preprocess >>>

  CHECK_CUDA(cudaDeviceSynchronize());
  CPU_TIMER_STOP(BFS_preprocess)
  tot_time += CPU_TIMER_ELAPSED(BFS_preprocess);
  CPU_TIMER_PRINT(BFS_preprocess)

  CPU_TIMER_INIT(BFS)


  //<<< kernel >>>



  CHECK_CUDA(cudaDeviceSynchronize());
  CPU_TIMER_STOP(BFS)
  tot_time += CPU_TIMER_ELAPSED(BFS);
  CPU_TIMER_PRINT(BFS)

  CPU_TIMER_INIT(BFS_postprocess)

 // <<< postprocess >>>

  CHECK_CUDA(cudaDeviceSynchronize());
  CPU_TIMER_STOP(BFS_postprocess)
  tot_time += CPU_TIMER_ELAPSED(BFS_postprocess);
  CPU_TIMER_PRINT(BFS_postprocess)

  // This output format is MANDATORY, DO NOT CHANGE IT
  printf("\n[OUT] Total BFS time: %f ms\n" RESET, tot_time);
*/
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

  CPU_TIMER_INIT(MTX_read)
  CSR_local<uint32_t, float> *csr =
      Distr_MMIO_CSR_local_read<uint32_t, float>(args.filename);
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
                     graph.col_idx, source, distances);
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
