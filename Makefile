CUDAC = nvcc
CUDA_FLAGS = -O3 -Iinclude -arch=sm_80
C_FLAGS = -Wall -Wextra# -std=c99
PROF_FLAGS = -lineinfo

SRC_DIR = src
BIN_DIR = bin

# SRCS = $(wildcard $(SRC_DIR)/{*.c,*.cpp,*.cu,*.cuh})
SRCS = $(SRC_DIR)/graph.cu $(SRC_DIR)/mmio.cu $(SRC_DIR)/bfs.cu

all: $(BIN_DIR)/bfs $(BIN_DIR)/bfs_profiling

$(BIN_DIR)/bfs: $(SRCS)
	@ mkdir -p $(BIN_DIR)
	$(CUDAC) $(CUDA_FLAGS) --compiler-options "$(C_FLAGS)" -o $@ $^

$(BIN_DIR)/bfs_profiling: $(SRCS)
	@ mkdir -p $(BIN_DIR)
	$(CUDAC) $(CUDA_FLAGS) --compiler-options "$(C_FLAGS)" $(PROF_FLAGS) -DENABLE_NVTX -o $@ $^

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
