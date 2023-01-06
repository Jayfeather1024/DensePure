#!/usr/bin/env bash

sigma=$1
steps=$2
majority_vote_num=$3

python merge_results.py \
--sample_id_list $(seq -s ' ' 0 500 49500) \
--sample_num 100 \
--majority_vote_num $majority_vote_num \
--N 10000 \
--N0 100 \
--sigma $sigma \
--classes_num 1000 \
--datasets imagenet \
--steps $steps