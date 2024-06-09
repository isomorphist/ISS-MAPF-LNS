#!/bin/bash


# Define directory path for storing error logs
dir_path_error=cluster_error

# Script to be submitted to the cluster
script_to_submit=lns.sh

# Job settings
time_limit=60
map_inst_path=ISS-MAPF-LNS/mapf_instances

# Possible maps

map_name = empty-8-8
# map_name = empty-32-32
# map_name = ost003d
# map_name=random-32-32-20
# map_name=warehouse-10-20-10-2-1


#modus=test # Modus: Modes for running the script (generateTrainData, testML, test)
neighbor_size=16
num_initial_solutions=30
num_LNS_runs=41


for modus in "testML" "test" 
do
sleep 1
for num_agents in $(seq 16 8 32)
do
sleep 1
    for scen_num in $(seq 1 1 25) # Loop through training scenarios or in test/testML number of tests through 25 test scenarios
    do
        # Construct a unique job name based on parameters
        job_name=mapf-$map_name-$modus-$(printf "%04d" $scen_num)-agents-$(printf "%03d" $num_agents)-num_isol-$num_initial_solutions-num_lns_runs-$num_LNS_runs-tlim-$time_limit-ns-$neighbor_size

        qsub -N $job_name \
             -l mem_free=8G \
             -l h_vmem=8G \
             -l bc5 \
             -r y \
             -e $dir_path_error/$job_name.out \
             -o /dev/null \
             $script_to_submit \
             $map_inst_path $map_name $modus $num_agents $scen_num $time_limit $neighbor_size $num_initial_solutions $num_LNS_runs

    done
done
done

