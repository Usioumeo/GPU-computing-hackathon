#!/bin/bash

source ~/.aliases
RESULTS_PATH=shared_test/gpu-computing-hackathon-results.json

while true; do
    baldo_get $RESULTS_PATH
    python3 gen_ranking.py baldo/$RESULTS_PATH
    sleep 30
done