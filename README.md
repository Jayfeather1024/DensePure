# DensePure: Understanding Diffusion Models towards Adversarial Robustness

<p align="center">
  <img width="1200" height="300" src="./pictures/densepure_flowchart.png">
</p>

Official PyTorch implementation of the paper:<br>
**[DensePure: Understanding Diffusion Models towards Adversarial Robustness](https://arxiv.org/abs/2211.00322)**
<br>
Chaowei Xiao, Zhongzhu Chen, Kun Jin, Jiongxiao Wang, Weili Nie, Mingyan Liu, Anima Anandkumar, Bo Li, Dawn Song<br>
https://densepure.github.io <br>

Abstract: *Diffusion models have been recently employed to improve certified robustness through the process of denoising. However, the theoretical understanding of why diffusion models are able to improve the certified robustness is still lacking, preventing from further improvement. In this study, we close this gap by analyzing the fundamental properties of diffusion models and establishing the conditions under which they can enhance certified robustness. This deeper understanding allows us to propose a new method DensePure, designed to improve the certified robustness of a pretrained model (i.e. classifier). Given an (adversarial) input, DensePure consists of multiple runs of denoising via the reverse process of the diffusion model (with different random seeds) to get multiple reversed samples, which are then passed through the classifier, followed by majority voting of inferred labels to make the final prediction. This design of using multiple runs of denoising is informed by our theoretical analysis of the conditional distribution of the reversed sample. Specifically, when the data density of a clean sample is high, its conditional density under the reverse process in a diffusion model is also high; thus sampling from the latter conditional distribution can purify the adversarial example and return the corresponding clean sample with a high probability. By using the highest density point in the conditional distribution as the reversed sample, we identify the robust region of a given instance under the diffusion model's reverse process. We show that this robust region is a union of multiple convex sets, and is potentially much larger than the robust regions identified in previous works. In practice, DensePure can approximate the label of the high density region in the conditional distribution so that it can enhance certified robustness. We conduct extensive experiments to demonstrate the effectiveness of DensePure by evaluating its certified robustness given a standard model via randomized smoothing. We show that DensePure is consistently better than existing methods on ImageNet, with 7% improvement on average.* 

## Requirements

- Python 3.8.5
- CUDA=11.1, PyTorch=1.8.0
- Installation of required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Datasets, pre-trained diffusion models and classifiers
Before running our code, you need to first prepare two datasets CIFAR-10 and ImageNet. CIFAR-10 will be downloaded automatically.
For ImageNet, you need to download validation images of ILSVRC2012 from https://www.image-net.org/. And the images need to be preprocessed by running the scripts `valprep.sh` from https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
under validation directory.  

Please change IMAGENET_DIR to your own location of ImageNet dataset in `datasets.py` before running the code.  

For the pre-trained diffusion models, you need to first download them from the following links:  
- [Improved Diffusion](https://github.com/openai/improved-diffusion) for
  CIFAR-10: (`cifar10_uncond_50M_500K.pt`: [download link](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt))
- [Guided Diffusion](https://github.com/openai/guided-diffusion) for
  ImageNet: (`256x256 diffusion unconditional`: [download link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt))

Please place all the pretrained models in the `pretrained` directory.  

For the pre-trained classifiers, they will be automatically downloaded by `timm` or `transformers`.  If you want to use your own
classifiers, code need to be changed in `eval_certified_densepure.py`.

## Run Experiments about Carlini 2022





## Run Experiments about DensePure



## License


## Citation
Please cite our paper and Carlini et al. (2022), if you happen to use this codebase:
```
@article{xiao2022densepure,
  title={DensePure: Understanding Diffusion Models towards Adversarial Robustness},
  author={Xiao, Chaowei and Chen, Zhongzhu and Jin, Kun and Wang, Jiongxiao and Nie, Weili and Liu, Mingyan and Anandkumar, Anima and Li, Bo and Song, Dawn},
  journal={arXiv preprint arXiv:2211.00322},
  year={2022}
}
```

```
@article{carlini2022certified,
  title={(Certified!!) Adversarial Robustness for Free!},
  author={Carlini, Nicholas and Tramer, Florian and Kolter, J Zico and others},
  journal={arXiv preprint arXiv:2206.10550},
  year={2022}
}
```


# run script example
'''
python eval_certified_densepure.py \
--exp exp \
--config cifar10.yml \
-i cifar10-certify_diffuse-ddpm-sample_num_100000-noise_0.50-2steps_$sample_id-$seed \
--domain cifar10 \
--seed 0 \
--data_seed 0 \
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
--predictions_path exp/0.50- \
--reverse_seed 0
'''