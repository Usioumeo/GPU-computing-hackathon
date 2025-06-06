#!/bin/bash

# source ~/.aliases
# RESULTS_PATH=shared_test/gpu-computing-hackathon-results.json

while true; do
    # baldo_get $RESULTS_PATH
    # python3 gen_ranking.py baldo/$RESULTS_PATH
    curl http://thomhub.ddns.net:7700/data > gpu-computing-hackathon-results.json
    python3 gen_ranking.py gpu-computing-hackathon-results.json
    sleep 240
done