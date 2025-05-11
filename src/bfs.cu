// #define ENABLE_NVTX
#define ENABLE_CPU_BASELINE
#define ENABLE_CORRECTNESS_CHECK

#define EXIT_INCORRECT_DISTANCES 10

#include <stdio.h>
#include <cuda_runtime.h>

#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "../include/colors.h"
#include "../include/utils.cuh"
#include "../include/graph.h"
#include "../include/cli.hpp"
#include "../include/mt19937-64.hpp"
#include "../include/bfs_baseline.cuh"

void gpu_bfs(
  const uint32_t N,           // Number of veritices
  const uint32_t M,           // Number of edges
  const uint32_t *h_rowptr,   // Graph CSR rowptr
  const uint32_t *h_colidx,   // Graph CSR colidx
  const uint32_t source,      // Source veritex
  int *h_distances            // Write here your distances
) {
  /***********************
   * IMPLEMENT HERE YOUR CUDA BFS
   * *********************/

  // !! This is just a placeholder !!
  gpu_bfs_baseline(N, M, h_rowptr, h_colidx, source, h_distances, true);
}

int main(int argc, char **argv) {
  int return_code = EXIT_SUCCESS;
  Cli_Args args;
  init_cli();
  if (parse_args(argc, argv, &args) != 0) {
    return -1;
  }

  GraphCSR *graph = import_mtx(args.filename);
  if (graph == NULL) {
    printf("Failed to import graph from file [%s]\n", args.filename);
    return -1;
  }

  // print_graph_csr(graph);

  uint32_t *sources = generate_sources(graph, args.runs, graph->num_vertices, args.source);
  int *distances_gpu_baseline = (int *)malloc(graph->num_vertices * sizeof(int));
  int *distances = (int *)malloc(graph->num_vertices * sizeof(int));

  for (int source_i = 0; source_i < args.runs; source_i++) {
    uint32_t source = sources[source_i];
    printf(GREEN "\n-- BFS iteration #%u, source=%u --\n", source_i, source);

    // Run the BFS baseline
    gpu_bfs_baseline(graph->num_vertices, graph->num_edges, graph->row_ptr, graph->col_idx, source, distances_gpu_baseline, false);

    #ifdef ENABLE_NVTX
		  nvtxRangePushA("Total BFS");
    #endif
    gpu_bfs(graph->num_vertices, graph->num_edges, graph->row_ptr, graph->col_idx, source, distances);
    #ifdef ENABLE_NVTX
		  nvtxRangePop();
    #endif

    bool match = true;
    #ifdef ENABLE_CORRECTNESS_CHECK
      for (uint32_t i = 0; i < graph->num_vertices; ++i) {
        if (distances_gpu_baseline[i] != distances[i]) {
          printf("Mismatch at node %u: Baseline distance = %d, Your distance = %d\n", i, distances_gpu_baseline[i], distances[i]);
          match = false;
          break;
        }
      }
      if (match) {
        printf(BRIGHT_GREEN "Correctness OK\n" RESET);
      } else {
        printf(BRIGHT_RED "GPU and CPU BFS results do not match for source node %u.\n" RESET, source);
        return_code = EXIT_INCORRECT_DISTANCES;
      }
    #endif

    #ifdef ENABLE_CPU_BASELINE
      int cpu_distances[graph->num_vertices];

      CPU_TIMER_INIT(CPU_BFS)
      cpu_bfs_baseline(graph->num_vertices, graph->row_ptr, graph->col_idx, source, cpu_distances);
      CPU_TIMER_CLOSE(CPU_BFS)

      match = true;
      for (uint32_t i = 0; i < graph->num_vertices; ++i) {
        if (distances_gpu_baseline[i] != cpu_distances[i]) {
          printf("Mismatch at node %u: GPU distance = %d, CPU distance = %d\n", i, distances_gpu_baseline[i], cpu_distances[i]);
          match = false;
          break;
        }
      }
      if (match) {
        printf(BRIGHT_GREEN "[CPU] Correctness OK\n" RESET);
      } else {
        printf(BRIGHT_RED "GPU and CPU BFS results do not match for source node %u.\n" RESET, source);
        return_code = EXIT_INCORRECT_DISTANCES;
      }
    #endif
  }

  free(sources);
  free(graph->row_ptr);
  free(graph->col_idx);
  free(graph);
  free(distances_gpu_baseline);
  free(distances);

  return return_code;
}
