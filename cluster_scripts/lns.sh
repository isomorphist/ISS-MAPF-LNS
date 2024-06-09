#!/bin/bash


# Variable Assignments
map_inst_path=$1
map_name=$2
modus=$3
num_agents=$4
scen_nr=$5
time_limit=$6
neighbor_size=$7
num_initial_solutions=$8
num_LNS_runs=$9


map_file="$map_inst_path/mapf-map/$map_name.map"
output_dir="$map_name-k-$(printf "%03d" $num_agents)-ns-$(printf "%02d" $neighbor_size)-initsol-$(printf "%03d" $num_initial_solutions)-runs-$(printf "%03d" $num_LNS_runs)"

# Define Paths
path_ml_local="machine_learning"
path_ml_cluster="$TMPDIR/machine_learning"
path_results_local="$path_ml_local/results/$output_dir"
path_results_cluster="$path_ml_cluster/results/$output_dir"

# Create Directories
mkdir -p $path_ml_local $path_ml_cluster $path_results_local $path_results_cluster

# Check modus and proceed accordingly
if [ "$modus" == "generateTrainData" ]; then
    path_train_data_local="$path_ml_local/train_data/$output_dir"
    path_train_data_cluster="$path_ml_cluster/train_data/$output_dir"
    output_log_local="$path_ml_local/train_data_output_log/$output_dir"
    output_log_cluster="$path_ml_cluster/train_data_output_log/$output_dir"
    scen_path_cluster="$path_ml_cluster/train_data_instances/$output_dir"
    scen_path_local="$path_ml_local/train_data_instances/$output_dir"
    scen_file="$scen_path_cluster/train-$(printf "%04d" $scen_nr).scen"

    # Create necessary directories
    mkdir -p $path_train_data_local $path_train_data_cluster $output_log_local $output_log_cluster $scen_path_cluster $scen_path_local


    # Execute the command
    build/lns -m $map_file -a $scen_file -k $num_agents -t $time_limit -s 0 --solver=LNS --initAlgo=PP --replanAlgo=PP --neighborSize=$neighbor_size --initDestoryStrategy=Adaptive --maxIterations=100000000 --modus=$modus --pathTrainData=$path_train_data_cluster --numInitialSolutions=$num_initial_solutions --numLNSRuns=$num_LNS_runs >> "$output_log_cluster/train-$(printf "%04d" $scen_nr).scen"

else
    scen_file="$map_inst_path/mapf-scen-random/scen-random/$map_name-random-$scen_nr.scen"

    if [ "$modus" == "testML" ]; then
        ml_model_local="$path_results_local/trained_ml_model.txt"
        cp $ml_model_local $path_results_cluster/
        result_file="$path_results_cluster/testML-LNS-$(printf "%04d" $scen_nr)"
    else
        result_file="$path_results_cluster/test-LNS-$(printf "%04d" $scen_nr)"
    fi

    
    for i in $(seq 1 1 60); do
        build/lns -m $map_file -a $scen_file -k $num_agents -o $result_file -t $time_limit -s 0 --solver=LNS --initAlgo=PP --replanAlgo=PP --neighborSize=$neighbor_size --initDestoryStrategy=Adaptive --maxIterations=100000000 --modus=$modus --numInitialSolutions=$num_initial_solutions --numLNSRuns=1 --pathResults=$path_results_cluster
    done
fi


# Synchronize the directories
rsync -a $path_ml_cluster/ $path_ml_local

# Exit the script
exit