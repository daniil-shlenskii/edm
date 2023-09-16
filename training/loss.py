# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import torch.nn.functional as F
# import torchvision.transforms as T
from torch import Tensor
from torch_utils import persistence
from training.nn import LearnableTimesteps

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

def get_inception_v3_feature_extractor() -> torch.nn.Module:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    # model.avgpool = torch.nn.Identity()
    model.dropout = torch.nn.Identity()
    model.fc = torch.nn.Identity()
    model.eval()
    return model

def normalize(image: Tensor, mean: Tensor, std: Tensor):
    mean, std = map(
        lambda x: torch.tensor(x, device=image.device).view(1, -1, 1, 1),
        [mean, std]
    )
    assert image.ndim == 4 and mean.shape == (1, 3, 1, 1)
    return (image - mean) / std


@persistence.persistent_class
class LinearKID:
    def __init__(self, sampler, image_to_timesteps: LearnableTimesteps):
        self.sampler = sampler
        self.image_to_timesteps = image_to_timesteps
        self._feature_extractor = get_inception_v3_feature_extractor()
    
    def __call__(self, net, images, labels=None, augment_pipe=None):
        if next(self._feature_extractor.parameters()).device != images.device:
            self._feature_extractor.to(images.device)
        b_size = images.shape[0]
        latents = torch.randn_like(images)
        timesteps = self.image_to_timesteps(images)
        xq = self.encoder(images).view(b_size, -1)
        xp = self.encoder(self.sampler(net, latents, timesteps)).view(b_size, -1).to(xq.dtype)
        
        add1 = torch.einsum("is,js->ij", xp, xp)
        add1 = add1 - torch.diag(torch.diag(add1))

        add2 = torch.einsum("is,js->ij", xp, xq)

        # return add1.mean() - 2 * add2.mean()
        return add1, add2

    def encoder(self, image: Tensor):
        image = image - image.min()
        image = image / image.max()
        image = F.interpolate(image, size=(224, 224), mode="area")
        image = normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return self._feature_extractor(image)
