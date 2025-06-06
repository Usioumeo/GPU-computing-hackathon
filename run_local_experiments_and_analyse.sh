#!/bin/bash

# Always run this script from the root of the repo

SKIP_EXPERIMENTS=0

# Parse CLI flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-experiments)
            SKIP_EXPERIMENTS=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

download_via_ssh() {
    local host_alias=$1
    local ssh_host=$2
    local remote_path=$3

    if [[ -z "$ssh_host" || -z "$remote_path" || -z "$host_alias" ]]; then
        echo "Usage: download_via_ssh <host_alias> <ssh_host> <remote_path>"
        return 1
    fi

    # Create the local directory structure
    local base_dir=$(dirname "$remote_path")
    mkdir -p "$host_alias/$base_dir"

    # Download the file or folder
    rsync -avz --progress --update "$ssh_host:$remote_path" "$host_alias/$base_dir/"

    if [[ $? -eq 0 ]]; then
        echo "Download completed successfully."
    else
        echo "Failed to download the file or folder."
        return 1
    fi
}

download_via_ssh . "${UNITN_USER}@baldo.disi.unitn.it" /data/hackathon/datasets

source env.sh "./data/hackathon/datasets"

if [[ $SKIP_EXPERIMENTS -eq 0 ]]; then
    echo -e "${GRE}Building $BIN...${NC}"
    make clean $BIN

    mkdir -p results

    echo -e "${GRE}%% Running tests on all graphs %%${NC}"
    for gi in ${!ALL_GRAPHS[@]}; do
        graph=${ALL_GRAPHS[$gi]}
        echo "----- Testing '$(basename "${graph%.*}")' graph -----"
        out_file="results/res_$(basename "${graph%.*}").out"
        $BIN -f "$MTX_PATH/$graph" -n $ITERATIONS > $out_file
        echo "Return code: $?" | tee $out_file
    done
else
    echo "Skipping experiments as requested by --skip-experiments flag."
fi

python3 scripts/gather_results_from_sout_local.py