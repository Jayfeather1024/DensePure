#!/usr/bin/env bash
cd ..

sigma=$1

python eval_certified_densepure.py \
--exp exp \
--config imagenet.yml \
-i imagenet-carlini22-sample_num_10000-noise_$sigma-1step \
--domain imagenet \
--seed 0 \
--diffusion_type guided-ddpm \
--lp_norm L2 \
--outfile results/imagenet-carlini22-sample_num_10000-noise_$sigma-1step \
--sigma $sigma \
--N 10000 \
--N0 100 \
--certified_batch 16 \
--sample_id $(seq -s ' ' 0 500 49500) \
--use_id \
--certify_mode purify \
--advanced_classifier beit \
--use_one_step