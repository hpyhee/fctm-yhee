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

from typing import List

import torch
import torch.nn as nn
from torch import Tensor, nn

from fctm.libs.regitsters import register_ft_restoration
from fctm.utils.utils import load_state_dict_from_server

from ...libcommon.default import base


import numpy as np
import math
import os
import sys

path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

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
        self.header_path = f"{os.path.join(path,dataset)}.hrd" 

        self.n_basis      = 256
        self.max_n_basis  = 1024
        self.num_layers   = 4
        self.num_tile_x   = 16
        self.num_tile_y   = 16
        self.num_tiles = (self.num_tile_y, self.num_tile_x)

        # vcm enc, dec params
        if self.n_basis > 0 or self.mode == "train":
            self.n_features   = 21760
            self.max_n_basis  = 256
            self.num_chs      = [256, 256, 256, 256]

            self.trained_path = os.path.join(path, f"ft_restoration/mlt_{self.task}_player_{self.max_n_basis}x{self.n_features}_cfp.pt")
            sys.path.insert(0, os.path.join(path))
            if self.mode=="train":
                self.ipca = IPCA(self.max_n_basis, self.n_features)
            else:
                self.ipca = torch.load(self.trained_path, map_location='cpu')

            self.ipca.to(device)
            self.ipca.prepare(self.n_basis)
            self.down = [ torch.nn.PixelUnshuffle(8), torch.nn.PixelUnshuffle(4), torch.nn.PixelUnshuffle(2), Dummy()]
            self.up   = [ torch.nn.PixelShuffle(8)  ,  torch.nn.PixelShuffle(4) , torch.nn.PixelShuffle(2)  , Dummy()]
                

    def decode(self, f_packed, ft_shapes, poc, file_prefix):
        self.tile_size = (ft_shapes[3][1],ft_shapes[3][2])
        # unpacking
        f_up  = torch.nn.functional.unfold(f_packed, self.tile_size, stride=self.tile_size)
        # inverse transform
        f_rec = self.ipca.invtransform(f_up)

        x = []
        ch_st = 0 
        for i in range(self.num_layers):
            ch_ed = ch_st + int(self.num_chs[i] * (2**((self.num_layers - 1 - i )*2)))
            r = self.up[i](f_rec.T[ch_st :ch_ed].reshape((1, -1, self.tile_size[0], self.tile_size[1])))
            ch_st = ch_ed
            x.append(r)
              
        x = sorted(x, key=lambda x: x.shape[-1], reverse=False)
        return x
    
@register_ft_restoration("m65704-o")
class prebm_invpca_o(base):
    """
    Pre-defined basis&mean + Energy Compensation Scaling
    m650705, Hannover, DE - October 2023
    Current implementation supports only for interfacing with R-CNN at FPN split points
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__(tag, device, **kwargs)
        split_coding_info = self.get_split_coding_info(tag, kwargs["task"], "dec")
        spif = split_coding_info["split_if"]
        self.spif = spif
        self.poc = 0
        print("PreBM decoder is loaded ...")
        self.mlt = MLT(device="cpu",task=kwargs["task"])


    @torch.no_grad()
    def forward(self, ftensors: Tensor, ft_shapes, file_prefix):
        d = self.to_device(ftensors)
        out = self.mlt.decode(d, ft_shapes, self.poc, file_prefix)
        self.poc=self.poc+1
        return self.sort_by_shape(out, ft_shapes)
