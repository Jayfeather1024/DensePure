# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random

import numpy as np

import torch
import torchvision.utils as tvu

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion import gaussian_diffusion

import math


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class Diffusion(torch.nn.Module):
    def __init__(self, args, config, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.reverse_state = None
        self.reverse_state_cuda = None

        print("Loading model")
        defaults = model_and_diffusion_defaults(self.args.t_total)
        model, diffusion = create_model_and_diffusion(**defaults)
        model.load_state_dict(
            dist_util.load_state_dict("pretrained/ddpm/cifar10_uncond_50M_500K.pt", map_location="cpu")
        )
        model.to(self.device)
        model.eval()

        self.model = model
        self.diffusion = diffusion
        sigma = self.args.sigma
        a = 1/(1+(sigma*2)**2)
        self.scale = a**0.5
        sigma = sigma*2
        T = self.args.t_total
        self.t = T*(1-(2*1.008*math.asin(math.sin(math.pi/(2*1.008))/(1+sigma**2)**0.5))/math.pi)


    def image_editing_sample(self, img=None, bs_id=0, tag=None, sigma=0.0):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]

        with torch.no_grad():
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            x0 = img

            x0 = self.scale*(img)
            t = self.t

            if self.args.use_clustering:
                x0 = x0.unsqueeze(1).repeat(1,self.args.clustering_batch,1,1,1).view(batch_size*self.args.clustering_batch,3,32,32)

            if self.args.use_one_step:
                # one step denoise
                t = torch.tensor([round(t)] * x0.shape[0], device=self.device)
                out = self.diffusion.p_sample(
                    self.model,
                    x0,
                    t+self.args.t_plus,
                    clip_denoised=True,
                )
                x0 = out["pred_xstart"]

            elif self.args.use_t_steps:

                #save random state
                if self.args.save_predictions:
                    global_seed_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        global_cuda_state = torch.cuda.random.get_rng_state_all()

                    if self.reverse_state==None:
                        torch.manual_seed(self.args.reverse_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(self.args.reverse_seed)
                    else:
                        torch.random.set_rng_state(self.reverse_state)
                        if torch.cuda.is_available():
                            torch.cuda.random.set_rng_state_all(self.reverse_state_cuda)

                # t steps denoise
                inter = t/self.args.num_t_steps
                indices_t_steps = [round(t-i*inter) for i in range(self.args.num_t_steps)]
                
                for i in range(len(indices_t_steps)):
                    t = torch.tensor([len(indices_t_steps)-i-1] * x0.shape[0], device=self.device)
                    real_t = torch.tensor([indices_t_steps[i]] * x0.shape[0], device=self.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            x0,
                            t,
                            clip_denoised=True,
                            indices_t_steps = indices_t_steps.copy(),
                            T = self.args.t_total,
                            step = len(indices_t_steps)-i,
                            real_t = real_t
                        )
                        x0 = out["sample"]

                #load random state
                if self.args.save_predictions:
                    self.reverse_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        self.reverse_state_cuda = torch.cuda.random.get_rng_state_all()

                    torch.random.set_rng_state(global_seed_state)
                    if torch.cuda.is_available():
                        torch.cuda.random.set_rng_state_all(global_cuda_state)

            else:
                #save random state
                if self.args.save_predictions:
                    global_seed_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        global_cuda_state = torch.cuda.random.get_rng_state_all()

                    if self.reverse_state==None:
                        torch.manual_seed(self.args.reverse_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(self.args.reverse_seed)
                    else:
                        torch.random.set_rng_state(self.reverse_state)
                        if torch.cuda.is_available():
                            torch.cuda.random.set_rng_state_all(self.reverse_state_cuda)

                # full steps denoise
                indices = list(range(round(t)))[::-1]
                for i in indices:
                    t = torch.tensor([i] * x0.shape[0], device=self.device)
                    with torch.no_grad():
                        out = self.diffusion.p_sample(
                            self.model,
                            x0,
                            t,
                            clip_denoised=True,
                        )
                        x0 = out["sample"]

                #load random state
                if self.args.save_predictions:
                    self.reverse_state = torch.random.get_rng_state()
                    if torch.cuda.is_available():
                        self.reverse_state_cuda = torch.cuda.random.get_rng_state_all()

                    torch.random.set_rng_state(global_seed_state)
                    if torch.cuda.is_available():
                        torch.cuda.random.set_rng_state_all(global_cuda_state)

            return x0
