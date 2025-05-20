#pragma once

#include "colors.h"
#include "graph.h"
#include "mt19937-64.hpp"
#include <cstdint>
#include <stdio.h>
#include <chrono>

/***** MACROS *****/
#define SEED 0

#define CEILING(x,y) (((x) + (y) - 1) / (y))

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%u\n", __FILE__, __LINE__); \
        exit(1); \
    }

#define CUDA_TIMER_DEF(name) \
    cudaEvent_t __timer_start_##name, __timer_stop_##name; \
    CHECK_CUDA(cudaEventCreate(&__timer_start_##name)); \
    CHECK_CUDA(cudaEventCreate(&__timer_stop_##name));

#define CUDA_TIMER_START(name) \
    CHECK_CUDA(cudaEventRecord(__timer_start_##name, 0));

#define CUDA_TIMER_STOP(name) \
    CHECK_CUDA(cudaEventRecord(__timer_stop_##name, 0)); \
    CHECK_CUDA(cudaEventSynchronize(__timer_stop_##name));

#define CUDA_TIMER_DESTROY(name) \
    CHECK_CUDA(cudaEventDestroy(__timer_start_##name)); \
    CHECK_CUDA(cudaEventDestroy(__timer_stop_##name));

#define CUDA_TIMER_ELAPSED(name) \
    ({ float elapsed_##name = 0.0f; \
       CHECK_CUDA(cudaEventElapsedTime(&elapsed_##name, __timer_start_##name, __timer_stop_##name)); \
       elapsed_##name; })

#define CUDA_TIMER_PRINT(name) \
    printf(BRIGHT_CYAN "Timer [%s] elapsed: %f ms\n" RESET, #name, CUDA_TIMER_ELAPSED(name));

#define CUDA_TIMER_INIT(name) CUDA_TIMER_DEF(name) CUDA_TIMER_START(name)

#define CUDA_TIMER_CLOSE(name) CUDA_TIMER_STOP(name) CUDA_TIMER_PRINT(name) CUDA_TIMER_DESTROY(name)

#define CPU_TIMER_DEF(name) \
  std::chrono::high_resolution_clock::time_point __timer_start_##name, __timer_stop_##name;

#define CPU_TIMER_START(name) \
  __timer_start_##name = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_STOP(name) \
  __timer_stop_##name = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_ELAPSED(name) \
  (std::chrono::duration<float>(__timer_stop_##name - __timer_start_##name).count()*1e3)

#define CPU_TIMER_PRINT(name) \
  printf(BRIGHT_CYAN "Timer [%s] elapsed: %f ms\n" RESET, #name, CPU_TIMER_ELAPSED(name));

#define CPU_TIMER_INIT(name) CPU_TIMER_DEF(name) CPU_TIMER_START(name)

#define CPU_TIMER_CLOSE(name) CPU_TIMER_STOP(name) CPU_TIMER_PRINT(name)

/***** BFS *****/
uint32_t *generate_sources(const GraphCSR *graph, int runs, uint32_t num_vertices, uint32_t source) {
  uint32_t *sources = (uint32_t*) malloc(runs * sizeof(uint32_t));
  if (source != UINT32_MAX) {
    for (int i = 0; i < runs; i++) {
      sources[i] = source;
    }
  } else {
    init_genrand64(SEED);
    for (int i = 0; i < runs; i++) {
      do {
        uint64_t gen = genrand64_int64();
        sources[i] = (uint32_t)gen % num_vertices;
      } while (graph->row_ptr[sources[i] + 1] - graph->row_ptr[sources[i]] == 0);
    }
  }
  return sources;
}