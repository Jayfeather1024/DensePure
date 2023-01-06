#!/usr/bin/env bash
cd ..

sigma=$1

python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--domain cifar10 \
--seed 0 \
--diffusion_type ddpm \
--lp_norm L2 \
--outfile results/cifar10-carlini22-sample_num_100000-noise_$sigma-1step \
--sigma $sigma \
--N 100000 \
--N0 100 \
--certified_batch 100 \
--sample_id $(seq -s ' ' 0 20 9980) \
--use_id \
--certify_mode purify \
--advanced_classifier vit \
--use_one_step