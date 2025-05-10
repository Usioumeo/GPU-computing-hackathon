# GPU Computing 2025 BFS Hackathon

This is the base repository for the hackathon. It contains:
* A baseline CUDA implementation of the Breadth Search First (BFS) algorithm, which includes:
    * Reading a MatrixMarket (.mtx) file
    * Creating the CSR representation of the graph
    * Executing the agorithm
* This baseline will be used for correctness checks
* An automated testing framework based on pre-defined datasets
* The code also includes:
    * Timers to record performance
    * NVTX examples

## Setup and Build

Compiling the project is as easy as it can get:

```bash
ml gcc........ cuda-11.3
make all # This will make targets "bin/bfs" "bin/bfs_profiling"
```

The `bfs_profiling` target will enable NVTX. 

## Code

You can start working directly on `src/bfs.cu`

## Datasets

