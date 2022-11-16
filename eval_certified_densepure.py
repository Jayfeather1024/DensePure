# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
from time import time
import datetime
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes, get_normalize_layer
import utils
from utils import str2bool, get_accuracy, get_image_classifier_certified, load_data
from runners.diffpure_ddpm_densepure import Diffusion
from runners.diffpure_guided_densepure import GuidedDiffusion
import torchvision.utils as tvu
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import timm
from networks import *

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class SDE_Adv_Model_certified_clustering(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        if args.domain == 'cifar10':
            if args.advanced_classifier=='vit':
                self.extractor = AutoFeatureExtractor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
                self.classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10").cuda()
            elif args.advanced_classifier=='cifar-wrn':
                self.classifier = Wide_ResNet(28,10,0.3,10)
                checkpoint = torch.load('../wide-resnet.pytorch/checkpoint/cifar10/wide-resnet-28x10.t7')
                self.classifier = checkpoint['net'].cuda()
                self.classifier.training = False
            else:
                self.classifier = get_image_classifier_certified('../models/cifar10/resnet110/noise_'+args.classifier_sigma+'/checkpoint.pth.tar', args.domain).to(config.device)
        elif args.domain == 'imagenet':
            if args.advanced_classifier=='beit':
                self.classifier = timm.create_model('beit_large_patch16_512', checkpoint_path='pretrained/beit_large_patch16_512_pt22k_ft22kto1k.pth').cuda()
                self.classifier.eval()
            elif args.advanced_classifier=='WRN':
                self.classifier = timm.create_model('wide_resnet50_2', pretrained=True).cuda()
                self.classifier.eval()
            elif args.advanced_classifier=='MLP':
                self.classifier = timm.create_model('mixer_b16_224_miil', pretrained=True).cuda()
                self.classifier.eval()
            elif args.advanced_classifier=='resnet':
                self.classifier = timm.create_model('resnet152', pretrained=True).cuda()
                self.classifier.eval()
            else:
                self.classifier = get_image_classifier_certified('../models/imagenet/resnet50/noise_'+args.classifier_sigma+'/checkpoint.pth.tar', args.domain).to(config.device)
        else:
            raise NotImplementedError('no classifier')        

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'guided-ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x, sample_id):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        start_time = time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag, sigma=self.args.sigma)
        minutes, seconds = divmod(time() - start_time, 60)

        # if self.args.save_info:
        #     np.save(self.args.image_folder+'/'+str(sample_id)+'-'+str(counter)+'-img_after_purify.npy',x_re.clone().detach().cpu().numpy())

        if 'imagenet' in self.args.domain:
            if self.args.advanced_classifier=='beit':
                x_re = F.interpolate(x_re, size=(512, 512), mode='bicubic')
            else:
                x_re = F.interpolate(x_re, size=(224, 224), mode='bicubic')

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        if self.args.advanced_classifier=='vit':
            self.classifier.eval()
            x_re = ((x_re+1)*0.5).cpu().numpy()
            x_re = [x for x in x_re]
            
            if self.args.vit_batch!=0:
                total_x_re_num = len(x_re)
                batch_num = math.ceil(total_x_re_num/self.args.vit_batch)
                out = torch.zeros(total_x_re_num, get_num_classes(self.args.domain))
                for i in range(batch_num):
                    if i==(batch_num-1):
                        inputs = self.extractor(x_re[i*self.args.vit_batch:], return_tensors="pt")
                        out[i*self.args.vit_batch:] = self.classifier(inputs["pixel_values"].cuda()).logits
                    else:
                        inputs = self.extractor(x_re[i*self.args.vit_batch:(i+1)*self.args.vit_batch], return_tensors="pt")
                        out[i*self.args.vit_batch:(i+1)*self.args.vit_batch] = self.classifier(inputs["pixel_values"].cuda()).logits
            else:
                inputs = self.extractor(x_re, return_tensors="pt")
                out = self.classifier(inputs["pixel_values"].cuda()).logits

        elif self.args.advanced_classifier=='cifar-wrn':
            self.classifier.eval()
            x_re = (x_re+1)*0.5
            means = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
            sds = torch.tensor([0.2471, 0.2435, 0.2616]).cuda()
            (batch_size, num_channels, height, width) = x_re.shape
            means = means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
            sds = sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
            x_re = (x_re - means)/sds
            out = self.classifier(x_re)

        elif self.args.advanced_classifier=='beit':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(x_re)

        elif self.args.advanced_classifier=='resnet':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(x_re)

        elif self.args.advanced_classifier=='WRN':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(x_re)

        elif self.args.advanced_classifier=='MLP':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(x_re)

        else:
            self.classifier.eval()
            out = self.classifier((x_re + 1) * 0.5)

        # if self.args.save_info:
        #     np.save(self.args.image_folder+'/'+str(sample_id)+'-'+str(counter)+'-logits.npy',out.clone().detach().cpu().numpy())

        self.counter += 1

        return out


class Certify_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        if args.domain == 'cifar10':
            if args.advanced_classifier=='vit':
                self.extractor = AutoFeatureExtractor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
                self.classifier = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10").cuda()
            elif args.advanced_classifier=='cifar-wrn':
                self.classifier = Wide_ResNet(28,10,0.3,10)
                checkpoint = torch.load('../wide-resnet.pytorch/checkpoint/cifar10/wide-resnet-28x10.t7')
                self.classifier = checkpoint['net'].cuda()
                self.classifier.training = False
            else:
                self.classifier = get_image_classifier_certified('../models/cifar10/resnet110/noise_'+args.classifier_sigma+'/checkpoint.pth.tar', args.domain).to(config.device)
        elif args.domain == 'imagenet':
            if args.advanced_classifier=='beit':
                self.classifier = timm.create_model('beit_large_patch16_512', checkpoint_path='pretrained/beit_large_patch16_512_pt22k_ft22kto1k.pth').cuda()
                self.classifier.eval()
            elif args.advanced_classifier=='WRN':
                self.classifier = timm.create_model('wide_resnet50_2', pretrained=True).cuda()
                self.classifier.eval()
            elif args.advanced_classifier=='MLP':
                self.classifier = timm.create_model('mixer_b16_224_miil', pretrained=True).cuda()
                self.classifier.eval()
            elif args.advanced_classifier=='resnet':
                self.classifier = timm.create_model('resnet152', pretrained=True).cuda()
                self.classifier.eval()
            else:
                self.classifier = get_image_classifier_certified('../models/imagenet/resnet50/noise_'+args.classifier_sigma+'/checkpoint.pth.tar', args.domain).to(config.device)
        else:
            raise NotImplementedError('no classifier')

    def forward(self, x, sample_id):

        if 'imagenet' in self.args.domain:
            if self.args.advanced_classifier=='beit':
                x = F.interpolate(x, size=(512, 512), mode='bicubic')
            else:
                x = F.interpolate(x, size=(224, 224), mode='bicubic')

        if self.args.advanced_classifier=='vit':
            self.classifier.eval()
            x_re = x.cpu().numpy()
            x_re = [x for x in x_re]
            
            if self.args.vit_batch!=0:
                total_x_re_num = len(x_re)
                batch_num = math.ceil(total_x_re_num/self.args.vit_batch)
                out = torch.zeros(total_x_re_num, get_num_classes(self.args.domain))
                for i in range(batch_num):
                    if i==(batch_num-1):
                        inputs = self.extractor(x_re[i*self.args.vit_batch:], return_tensors="pt")
                        out[i*self.args.vit_batch:] = self.classifier(inputs["pixel_values"].cuda()).logits
                    else:
                        inputs = self.extractor(x_re[i*self.args.vit_batch:(i+1)*self.args.vit_batch], return_tensors="pt")
                        out[i*self.args.vit_batch:(i+1)*self.args.vit_batch] = self.classifier(inputs["pixel_values"].cuda()).logits
            else:
                inputs = self.extractor(x_re, return_tensors="pt")
                out = self.classifier(inputs["pixel_values"].cuda()).logits

        elif self.args.advanced_classifier=='cifar-wrn':
            self.classifier.eval()
            x_re = x
            means = torch.tensor([0.4914, 0.4822, 0.4465]).cuda()
            sds = torch.tensor([0.2471, 0.2435, 0.2616]).cuda()
            (batch_size, num_channels, height, width) = x_re.shape
            means = means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
            sds = sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
            x_re = (x_re - means)/sds
            out = self.classifier(x_re)

        elif self.args.advanced_classifier=='beit':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(2*x-1)

        elif self.args.advanced_classifier=='resnet':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(2*x-1)

        elif self.args.advanced_classifier=='WRN':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(2*x-1)

        elif self.args.advanced_classifier=='MLP':
            with torch.no_grad():
                self.classifier.eval()
                out = self.classifier(2*x-1)

        else:
            self.classifier.eval()
            out = self.classifier(x)

        return out


def original_certify(dataset, args, config):
    # ---------------- evaluate certified robustness of classifier/smoothed classifier ----------------

    ngpus = torch.cuda.device_count()

    classifier = Certify_Model(args, config)
    if ngpus > 1:
        classifier = torch.nn.DataParallel(classifier)
    classifier = classifier.eval().to(config.device)
    print(f'evaluate certified robustness of classifier [{args.lp_norm}]...')
    smoothed_classifier = Smooth(classifier, get_num_classes(args.domain), args.sigma)
    
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    if args.use_id:
        for i in args.sample_id:
            (x, label) = dataset[i]

            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            label = torch.tensor(label,dtype=torch.int).cuda()
            prediction, radius, n0_predictions, n_predictions = smoothed_classifier.certify(x, args.N0, args.N, i, args.alpha, args.certified_batch, args.clustering_method)
            after_time = time()
            correct = int(prediction == label)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        f.close()

    else:
        for i in range(len(dataset)):
            # only certify every args.skip examples, and stop after args.max examples
            if i % args.skip != 0:
                continue
            if i == args.max:
                break
            (x, label) = dataset[i]

            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            label = torch.tensor(label,dtype=torch.int).cuda()
            prediction, radius, n0_predictions, n_predictions = smoothed_classifier.certify(x, args.N0, args.N, i, args.alpha, args.certified_batch, args.clustering_method)
            after_time = time()
            correct = int(prediction == label)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        f.close()        

def purified_certify(model, dataset, args, config):
    # ---------------- evaluate certified robustness of diffpure + classifier ----------------
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    print(f'apply the attack to diffpure + classifier [{args.lp_norm}]...')
    model_.reset_counter()
    smoothed_classifier_diffpure = Smooth(model, get_num_classes(args.domain), args.sigma)
    f = open(args.outfile+'_diffpure', 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    if args.use_id:
        for i in args.sample_id:
            (x, label) = dataset[i]

            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            label = torch.tensor(label,dtype=torch.int).cuda()
            prediction, radius, n0_predictions, n_predictions = smoothed_classifier_diffpure.certify(x, args.N0, args.N, i, args.alpha, args.certified_batch, args.clustering_method)
            after_time = time()
            correct = int(prediction == label)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
            if args.save_predictions:
                np.save(args.predictions_path+str(i)+'-'+str(args.reverse_seed)+'-n0_predictions.npy',n0_predictions)
                np.save(args.predictions_path+str(i)+'-'+str(args.reverse_seed)+'-n_predictions.npy',n_predictions)
        f.close()

    else:
        for i in range(len(dataset)):

            # only certify every args.skip examples, and stop after args.max examples
            if i % args.skip != 0:
                continue
            if i == args.max:
                break
            (x, label) = dataset[i]

            before_time = time()
            # certify the prediction of g around x
            x = x.cuda()
            label = torch.tensor(label,dtype=torch.int).cuda()
            prediction, radius, n0_predictions, n_predictions = smoothed_classifier_diffpure.certify(x, args.N0, args.N, i, args.alpha, args.certified_batch, args.clustering_method)
            after_time = time()
            correct = int(prediction == label)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
            if args.save_predictions:
                np.save(args.predictions_path+str(i)+'-'+str(args.reverse_seed)+'-n0_predictions.npy',n0_predictions)
                np.save(args.predictions_path+str(i)+'-'+str(args.reverse_seed)+'-n_predictions.npy',n_predictions)
        f.close()


def robustness_eval(args, config):
    log_dir = os.path.join(args.image_folder, 'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model_certified_clustering(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)

    # load dataset
    dataset = get_dataset(args.domain, 'test')

    # eval classifier and sde_adv against attacks
    if args.certify_mode == 'both':
        original_certify(dataset, args, config)
        purified_certify(model, dataset, args, config)
    elif args.certify_mode == 'purify':
        purified_certify(model, dataset, args, config)
    elif args.certify_mode == 'base':
        original_certify(dataset, args, config)
    else:
        raise NotImplementedError('unknown certify mode')

    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # dataset
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')

    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])

    # certified robustness
    parser.add_argument('--sigma', type=float, default=0.5, help='noise hyperparameter')
    parser.add_argument('--classifier_sigma', type=str, default=0.00, help='sigma for choosing classifier')
    parser.add_argument("--skip", type=int, default=100, help="how many examples to skip")
    parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
    parser.add_argument("--N0", type=int, default=100)
    parser.add_argument("--N", type=int, default=5000, help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument("--certified_batch", type=int, default=400, help="batch size")
    parser.add_argument("--outfile", type=str, default='results/test5000', help="output file")
    parser.add_argument('--use_id', action='store_true', help='evaluate specific sample')
    parser.add_argument("--sample_id", type=int, nargs='+', default=[0], help="sample id for evaluation")
    parser.add_argument("--certify_mode", type=str, default="purify", help="base, purify or both")
    parser.add_argument("--advanced_classifier", type=str, default="none", help="vit")
    parser.add_argument("--vit_batch", type=int, default=0, help="batch size")
    parser.add_argument('--use_one_step', action='store_true', help='whether to use one step denoise')
    parser.add_argument('--use_parallel', action='store_true', help='whether to use multi gpus to compute radius')
    parser.add_argument('--save_predictions', action='store_true', help='whether to save predictions')
    parser.add_argument("--predictions_path", type=str, default='../npy', help="npy save file")
    parser.add_argument('--reverse_seed', type=int, default=0, help='reverse seed')
    parser.add_argument('--use_t_steps', action='store_true', help='whether to use t steps denoise')
    parser.add_argument('--num_t_steps', type=int, default=1, help='numbers of reverse t steps')
    parser.add_argument('--t_plus', type=int, default=0, help='perturbation of t')
    parser.add_argument('--t_total', type=int, default=4000, help='total t to reduce reverse t')
    parser.add_argument('--save_info', action='store_true', help='whether to save image logits')
    parser.add_argument('--use_clustering', action='store_true', help='whether to use clustering when purifying')
    parser.add_argument('--clustering_batch', type=int, default=100)
    parser.add_argument("--clustering_method", type=str, default="none", help="classifier")

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    print(args)
    robustness_eval(args, config)