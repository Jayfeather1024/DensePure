#!/usr/bin/env bash
cd ..

seed=$s1

python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-densepure-sample_num_100000-noise_0.25-2steps_$sample_id-$seed \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile results/cifar10-certify_diffse-ddpm-noise_0.50-sample_100000-2steps_$sample_id-$seed \
--sigma 0.25 \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id 0 \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_t_steps \
--num_t_steps 2 \
--save_predictions \
--predictions_path exp/0.25- \
--reverse_seed $seed