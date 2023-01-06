#!/usr/bin/env bash

sigma=$1
steps=$2
majority_vote_num=$3

python merge_results.py \
--sample_id_list $(seq -s ' ' 0 20 9980) \
--sample_num 500 \
--majority_vote_num $majority_vote_num \
--N 100000 \
--N0 100 \
--sigma $sigma \
--classes_num 10 \
--datasets cifar10 \
--steps $steps
