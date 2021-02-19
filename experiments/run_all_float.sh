#!/bin/bash

SAMPLES=3
ret_val=""
GPU=$1
function run_experiments {
    declare -a samples_list=()
    for i in $(seq 1 $SAMPLES);
    do 
        local sample=$(python3 $1 --seed $i --gpu $GPU | grep Experiment | cut -d " " -f 4)
        echo $sample
        samples_list+=("../${sample}")
    done
    echo "${samples_list[*]}"
    if [ ! -d "./summary" ]; then
        mkdir summary
    fi 
    cd ./summary/
    result=$(python3 ../$2 --result_paths "${samples_list[*]}" --label $3  | grep Experiment | cut -d " " -f 4)
    ret_val=$result
    cd ../
}

function run_q_experiments {
    local float_samples=(./../../float/default/*"$4"*)
    echo "${float_samples[*]}"
    declare -a samples_list=()
    for i in $(seq 1 $SAMPLES);
    do 
        local index=${i}-1
        echo "${float_samples[$index]}"
        local sample=$(python3 $1 --seed $i --gpu $GPU --load ${float_samples[$index]} | grep Experiment | cut -d " " -f 4)
        echo $sample
        samples_list+=("../${sample}")
    done
    if [ ! -d "./summary" ]; then
        mkdir summary
    fi 
    cd ./summary/
    echo "${samples_list[*]}"
    result=$(python3 ../$2 --result_paths "${samples_list[*]}" --label $3 | grep Experiment | cut -d " " -f 4)
    ret_val=$result
    cd ../
}

cd ./scripts/pointwise/float
#run_experiments pointwise_regression.py ../../../average_results.py float_pointwise_regression
#run_experiments pointwise_mnist.py ../../../average_results.py float_pointwise_mnist
#run_experiments pointwise_cifar.py ../../../average_results.py float_pointwise_cifar
cd ./../../../

cd ./scripts/stochastic/mcdropout/float
#run_experiments mcdropout_regression.py ../../../../average_results.py float_mcdropout_regression
#run_experiments mcdropout_mnist.py ../../../../average_results.py float_mcdropout_mnist
#run_experiments mcdropout_cifar.py ../../../../average_results.py float_mcdropout_cifar
cd ./../../../../

cd ./scripts/stochastic/sgld/float
#run_experiments sgld_regression.py ../../../../average_results.py float_sgld_regression
#run_experiments sgld_mnist.py ../../../../average_results.py float_sgld_mnist
#run_experiments sgld_cifar.py ../../../../average_results.py float_sgld_cifar
cd ./../../../../

cd ./scripts/stochastic/bbb/float
#run_experiments bbb_regression.py ../../../../average_results.py float_bbb_regression
#run_experiments bbb_mnist.py ../../../../average_results.py float_bbb_mnist
#run_experiments bbb_cifar.py ../../../../average_results.py float_bbb_cifar
cd ./../../../../
