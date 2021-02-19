#!/bin/bash

SAMPLES=3
ACTIVATION_PRECISION=7
WEIGHT_PRECISION=8
GPU=$1
function run_q_experiments {
    local float_samples=(./../../float/default/*"$4"*)
    echo "${float_samples[*]}"

    for w in $(seq 3 $WEIGHT_PRECISION);
    do
        echo "a_${ACTIVATION_PRECISION}_w_${w}"
        if [ ! -d "a_${ACTIVATION_PRECISION}_w_${w}" ]; then
            mkdir a_"${ACTIVATION_PRECISION}"_w_"${w}"
        fi
        cd ./a_"${ACTIVATION_PRECISION}"_w_"${w}"/
        declare -a samples_list=()
        for i in $(seq 1 $SAMPLES);
        do  
            local index=${i}-1
            echo "${float_samples[$index]}"
            local sample=$(python3 ../$1 --seed $i --gpu $GPU --load ../${float_samples[$index]} --weight_precision $w --activation_precision ${ACTIVATION_PRECISION} | grep Experiment | cut -d " " -f 4)
            echo $sample
            samples_list+=("../${sample}")
        done
        if [ ! -d "./summary" ]; then
            mkdir summary
        fi 
        cd ./summary/
        echo "${samples_list[*]}"
        result=$(python3 ../../$2 --result_paths "${samples_list[*]}" --label $3_${ACTIVATION_PRECISION}_${w} | grep Experiment | cut -d " " -f 4)
        ret_val=$result
        cd ../../
    done

    for a in $(seq 3 $(($ACTIVATION_PRECISION-1)));
    do
        echo "a_${a}_w_${WEIGHT_PRECISION}"
        if [ ! -d "a_${a}_w_${WEIGHT_PRECISION}" ]; then
            mkdir a_"${a}"_w_"${WEIGHT_PRECISION}"
        fi
        cd ./a_"${a}"_w_"${WEIGHT_PRECISION}"/
        declare -a samples_list=()
        for i in $(seq 1 $SAMPLES);
        do  
            local index=${i}-1
            echo "${float_samples[$index]}"
            local sample=$(python3 ../$1 --seed $i --gpu $GPU --load ../${float_samples[$index]} --weight_precision ${WEIGHT_PRECISION} --activation_precision $a | grep Experiment | cut -d " " -f 4)
            echo $sample
            samples_list+=("../${sample}")
        done
        if [ ! -d "./summary" ]; then
            mkdir summary
        fi 
        cd ./summary/
        echo "${samples_list[*]}"
        result=$(python3 ../../$2 --result_paths "${samples_list[*]}" --label $3_${a}_${WEIGHT_PRECISION} | grep Experiment | cut -d " " -f 4)
        ret_val=$result
        cd ../../
    done
}

cd ./scripts/pointwise/quantised/train
#run_q_experiments pointwise_regression.py ../../../../average_results.py qat_pointwise_regression regression-regression
#run_q_experiments pointwise_mnist.py ../../../../average_results.py qat_pointwise_mnist mnist
#run_q_experiments pointwise_cifar.py ../../../../average_results.py qat_pointwise_cifar cifar
cd ./../../../../

cd ./scripts/stochastic/mcdropout/quantised/train
#run_q_experiments mcdropout_regression.py ../../../../../average_results.py qat_mcdropout_regression regression-regression
#run_q_experiments mcdropout_mnist.py ../../../../../average_results.py qat_mcdropout_mnist mnist
#run_q_experiments mcdropout_cifar.py ../../../../../average_results.py qat_mcdropout_cifar cifar
cd ./../../../../../

cd ./scripts/stochastic/sgld/quantised/train
#run_q_experiments sgld_regression.py ../../../../../average_results.py qat_sgld_regression regression-regression
#run_q_experiments sgld_mnist.py ../../../../../average_results.py qat_sgld_mnist mnist
#run_q_experiments sgld_cifar.py ../../../../../average_results.py qat_sgld_cifar cifar
cd ./../../../../../

cd ./scripts/stochastic/bbb/quantised/train
#run_q_experiments bbb_regression.py ../../../../../average_results.py qat_bbb_regression regression-regression
#run_q_experiments bbb_mnist.py ../../../../../average_results.py qat_bbb_mnist mnist
#run_q_experiments bbb_cifar.py ../../../../../average_results.py qat_bbb_cifar cifar
