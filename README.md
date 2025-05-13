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

First, on you **local machine** run:

```bash
git submodule init
git submodule update
```

Then set the `UNITN_USER` variable and sync the local repository on `baldo`:

```bash
export UNITN_USER=<name.surname>
./baldo_sync.sh # This requires 'rsync'
```

On `baldo', compiling the project is as easy as it can get:

```bash
ml CUDA/12.5.0
make all # This will make targets "bin/bfs" "bin/bfs_profiling"
```

The `bfs_profiling` target will enable NVTX. 

## Code

You can start working directly on `src/bfs.cu`

## Running Experiments

> **_IMPORTANT:_** First, set the name of you group in the `env.sh` file.

Please use the following to run you experiments:

```bash
# Run this from the repo root folder 
./run_experiments.sh # [--no-small-diam] [--no-large-diam]
```

*The optional flags disable test for the given category of graphs*

Running the script will take care of setting the correct configuration for SLURM and submits one job per graph.

## Datasets

