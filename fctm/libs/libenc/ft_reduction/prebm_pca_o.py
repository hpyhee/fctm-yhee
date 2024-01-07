# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.

# Copyright (c) 2023, ISO/IEC
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the ISO/IEC nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.


import fvcore.nn.weight_init as weight_init  # fvcore==0.1.5.post20210812
import torch
from torch import nn
from fctm.libs.regitsters import register_ft_reduction
from fctm.utils.utils import load_state_dict_from_server
import numpy as np
import math
import os
import sys

path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
from ...libcommon.default import base

class IPCA:
    def __init__(self, n_components, n_features) -> None:
        self.n_components   = n_components
        self.n_features     = n_features
        self.n_samples_seen = 0
        self.step           = 0
        self.explained_variance = None
        self.noise_variance     = None
        self.running_mean  = torch.zeros((1, n_features)           , dtype=torch.float32)
        self.components    = torch.zeros((n_components, n_features), dtype=torch.float32)
        self.singular_vals = torch.zeros((n_components)            , dtype=torch.float32)

    def to(self, device):
        self.running_mean  = self.running_mean.to(device)
        self.components    = self.components.to(device)
        return self
    
    def prepare(self, n_components, enc=True):
        self.W = self.components[:n_components].T
        if enc:
            self.B = self.running_mean @ self.components[:n_components].T

    def fit(self, inputs):
        n_samples, n_features = inputs.shape
        assert n_features == self.n_features, f"{n_features} != {self.n_features}"
        
        col_batch_mean = torch.mean(inputs, -2, keepdim=True)

        n_total_samples = self.n_samples_seen + n_samples
        col_mean = self.running_mean * self.n_samples_seen
        col_mean += torch.sum(inputs, -2, keepdim=True)
        col_mean /= n_total_samples

        mean_correction = math.sqrt((self.n_samples_seen * n_samples) / (self.n_samples_seen + n_samples)) * (self.running_mean - col_batch_mean)

        x = inputs - col_batch_mean
        if self.n_samples_seen != 0:
            x = torch.concat(
                [
                    torch.reshape(self.singular_vals, [-1, 1]) * self.components,
                    inputs - col_batch_mean,
                    mean_correction,
                ],
                dim=0,
            )
        _, s, v = torch.svd(x)
        v = v.T
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        v *= signs.reshape((-1,1))

        self.explained_variance = torch.square(s) / (n_total_samples - 1)
        self.noise_variance     = torch.mean(self.explained_variance[self.n_components:])
        self.components      = v[:self.n_components]
        self.singular_vals   = s[:self.n_components]
        self.running_mean    = col_mean
        self.n_samples_seen += n_samples
        self.step           += 1 
    
    def transform(self, inputs):
        return (inputs @ self.W) - self.B
    
    def invtransform(self, inputs):
        return (inputs @ self.W.T) + self.running_mean

class Dummy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
    
class MLT:
    def __init__(self, task="", device="cuda" ) -> None:
        self.device = device
        self.task  = task
        self.mode = "test"
        dataset= "dataset"
        
        self.n_basis      = 256
        self.max_n_basis  = 1024
        self.num_layers   = 4
        self.num_tile_x   = 16
        self.num_tile_y   = 16

        self.num_tiles = (self.num_tile_y, self.num_tile_x)

        # vcm enc, dec params
        if self.n_basis > 0 or self.mode == "train":
            self.w  = [1, 1, 1, 1]
            self.n_features   = 21760
            self.max_n_basis  = 256
            self.num_chs      = [256, 256, 256, 256]

            self.trained_path = os.path.join(path, f"ft_reduction/mlt_{self.task}_player_{self.max_n_basis}x{self.n_features}_cfp.pt")
            sys.path.insert(0, os.path.join(path))
            if self.mode=="train":
                self.ipca = IPCA(self.max_n_basis, self.n_features)
            else:
                self.ipca = torch.load(self.trained_path, map_location='cpu')

            self.ipca.to(device)
            self.ipca.prepare(self.n_basis)
            self.down = [ torch.nn.PixelUnshuffle(8), torch.nn.PixelUnshuffle(4), torch.nn.PixelUnshuffle(2), Dummy()]
            self.up   = [ torch.nn.PixelShuffle(8)  ,  torch.nn.PixelShuffle(4) , torch.nn.PixelShuffle(2)  , Dummy()]
                
    def encode(self, x, result_dir):
        
        x = sorted(x, key=lambda x: x.shape[-1], reverse=True)
        self.tile_size = x[-1].shape[2:]

        # feature reshaping & merge
        x = torch.cat([ down(f) for f, down in zip(x, self.down)], dim=1)
        x = x.reshape(( x.shape[1], x.shape[2]*x.shape[3]))
        x = x.transpose(0, 1)

        # PCA forward transform 
        f = self.ipca.transform(x)
        self.packed_size = (self.tile_size[0] * self.num_tiles[0], self.tile_size[1] * self.num_tiles[1])
        f_p        = torch.nn.functional.fold(f.unsqueeze(0), self.packed_size, self.tile_size, stride=self.tile_size)

        return f_p

@register_ft_reduction("m65704-o")
class prebm_pca_o(base):
    """
    Pre-defined basis&mean + (w/o Energy Compensation Scaling)
    m650705, Hannover, DE - October 2023
    Current implementation supports only for interfacing with R-CNN at FPN split points
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__(tag, device, **kwargs)

        split_coding_info = self.get_split_coding_info(tag, kwargs["task"], "enc")
        spif = split_coding_info["split_if"]        
        print("PreBM encoder is loaded ...")
        self.mlt = MLT(device="cpu",task=kwargs["task"])

    @torch.no_grad()
    def forward(self, ftensors,result_dir):
        
        data = [self.to_device(v.unsqueeze(0)) for v in ftensors.values()][::-1]
        out = self.mlt.encode(data, result_dir)
        # out = torch.from_numpy(self.mlt.encode(data))

        return out.squeeze(dim=1)
