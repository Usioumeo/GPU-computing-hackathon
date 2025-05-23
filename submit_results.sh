#!/bin/bash

source env.sh

if [[ $GROUP_NAME == "test_group" ]]; then
    echo -e "${RED} Please set your group name in 'group_name.sh'${NC}"
    exit 1
fi

source SbatchMan/sourceFile.sh
queue=$(${SbM_UTILS}/inQueue.sh)
go=1

if [[ ! -z $queue ]]; then
    echo -e "-- QUEUE --\n${queue}"
    echo -e "${PUR}Queue is not empty. Do you want to proceed anyway? (y/N)${NC}"
    read -r response
    if [[ "$response" != "y" && "$response" != "Y" ]]; then
        exit
    fi
fi

running=$(${SbM_UTILS}/whoRuns.sh)

if [[ ! -z $running ]]; then
    echo -e "-- RUNNING --\n${running}"
    echo -e "${PUR}Jobs are still running. Do you want to proceed anyway? (y/N)${NC}"
    read -r response
    if [[ "$response" != "y" && "$response" != "Y" ]]; then
        exit
    fi
fi

echo -e "${GRE}Submitting results...${NC}"
python3 scripts/gather_results_from_sout.py
ret=$?
if [[ $ret -eq 0 ]]; then
    res=$(python3 scripts/gather_results_from_sout.py --submission)
    # echo "---------------------------------------"
    # echo "$res"
    curl -X POST http://thomhub.ddns.net:7700/append -H "Content-Type: application/json" -d "$res"
    # shared_file="$SHARED_DIR/gpu-computing-hackathon-results.json"
    # # if [[ ! -f $shared_file ]]; then
    # touch $shared_file
    # # fi
    # printf "%s\n" "$res" >> $shared_file
else
    echo -e "${RED}Something went wrong${NC} (return code: $ret)"
    # printf "%s\n" "$res"
fi
