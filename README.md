# GPU Computing 2025 BFS Hackathon

This is the base repository for the hackathon. It contains:
* A baseline CUDA implementation of the Breadth Search First (BFS) algorithm, which includes:
    * Reading a MatrixMarket (.mtx) file
    * Creating the CSR representation of the graph
    * Executing the algorithm
* This baseline will be used for correctness checks
* An automated testing framework based on pre-defined datasets
* The code also includes:
    * Timers to track performance
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

On `baldo`, compiling the project is as easy as it can get:

```bash
ml CUDA/12.5.0
make all # This will make targets "bin/bfs" "bin/bfs_dbg" "bin/bfs_profiling"
```

The `bfs_profiling` target will enable NVTX. The `bfs_dbg` target will set the `DEBUG_PRINTS` preprocessor variable, enabling debug prints. *Feel free to change this as needed.*

> **IMPORTANT:** only the target `bin/bfs` will be used in the performance tests.

<!-- ### Downloading Datasets (if necessary)

First, on you system, from the repo root, run:

```bash
cd MtxMan
git submodule init
git submodule update
cd ..
./baldo_sync.sh
```

Then on `baldo`, from the repo root, run:
```bash
cp config.yaml MtxMan/config.yaml
python3 -m venv .venv
source .venv/bin/activate
cd MtxMan
pip install -r requirements.txt
python3 scripts/sync_datasets.py --binary-mtx
``` -->

## Code

You can start working directly on `src/bfs.cu`

## Environment

> **_IMPORTANT:_** First, set the name of you group in the `env.sh` file.

<!-- On each terminal you open in the cluster, make sure to first run:

```bash
source env.sh
``` -->

## Running Experiments

Please use the following to run you experiments:

```bash
# Run this from the repo root folder 
./run_experiments.sh # [--no-small-diam] [--no-large-diam] [--no-graph500]
```

*The optional flags disable test for the given category of graphs*

Running the script will take care of setting the correct configuration for SLURM and submits one job per graph.

### Local Testing

To run test on your (local) machine, run:

```bash
./run_local_experiments_and_analyse.sh # [--skip-experiments] this just analyses the results
```

This will:
- Download the graphs (4GB disk space required)
- Run `bin/bfs` on each graph
- Parse their stdout+stderr with `scripts/gather_results_from_sout_local.py`

Feel free to customize `scripts/gather_results_from_sout_local.py` as needed.

## Submitting Results

To submit the results:

```bash
# Run this from the repo root folder 
./submit_results.sh
```

*If experiments are still running, the script will notify it.*

## Datasets

The datasets are divided into three categories:
* Small-diameter
* Large-diameter
* Graph500 Kronecker 

Here is a summary of theirs characteristics:

| **Name**           | **N** | **M** | **Category**       | **Diameter**  |
|--------------------|-------|-------|--------------------|---------------|
| roadNet-CA         | 1M    |   5M  | Small-diameter     | 500-600       |
| wikipedia-20070206 | 3M    |  45M  | Small-diameter     | 459-460       |
| soc-LiveJournal1   | 5M    |  69M  | Small-diameter     | 14-16         |
| hollywood-2009     | 1M    | 114M  | Small-diameter     | 9-10          |
| GAP-road           | 23M   |  58M  | Large-diameter     | 5k-7k         |
| rgg_n_2_22_s0      | 4M    |  60M  | Large-diameter     | 1k-1.5k       |
| europe_osm         | 50M   | 108M  | Large-diameter     | 15k-28k       |
| rgg_n_2_24_s0      | 18M   | 265M  | Large-diameter     | 1.7k-2.7k     |
| graph500_20_8      | 1M    | 8M    | Graph500 Kronecker | 10-14         |
| graph500_21_8      | 2M    | 17M   | Graph500 Kronecker | 8-17          |
| graph500_20_32     | 1M    | 33M   | Graph500 Kronecker | 9-12          |
| graph500_21_16     | 2M    | 33M   | Graph500 Kronecker | 8-17          |

*The diameter varies depending on the source node*

<!-- 
| webbase-2001       | 118M  | 1B    | Small-diameter     |
| twitter7           | 42M   | 1.5B  | Small-diameter     |
| GAP-twitter        | 61M   | 1.5B  | Small-diameter     |
| GAP-web            | 50M   | 1.9B  | Small-diameter     | -->



## Ranking

The `submit_results` will automatically upload you results to a shared ranking list.

Some notes about how the ranking works:

* Only submissions with correct results for all graphs will be considered
* For each group, the submission with the highest global (across all datasets) speedup geomean is considered.
