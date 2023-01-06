#!/usr/bin/env bash
cd ..

reverse_seed=$s1
sigma=$2
steps=$3

python eval_certified_diffpure_clustering.py \
--exp exp \
--config imagenet.yml \
-i imagenet-densepure-sample_num_10000-noise_$sigma-$steps-$reverse_seed \
--domain imagenet \
--seed 0 \
--diffusion_type guided-ddpm \
--lp_norm L2 \
--outfile imagenet-densepure-sample_num_10000-noise_$sigma-$steps-$reverse_seed \
--sigma $sigma \
--N 10000 \
--N0 100 \
--certified_batch 16 \
--sample_id $(seq -s ' ' 0 500 49500) \
--use_id \
--certify_mode purify \
--advanced_classifier beit \
--use_t_steps \
--num_t_steps $steps \
--save_predictions \
--predictions_path exp/imagenet/$sigma- \
--reverse_seed $reverse_seed