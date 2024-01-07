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
    def __init__(self, split_point="eval", device="cuda" ) -> None:
        POC=0
        self.device = device
        self.split_point  = split_point
        self.mode = "test"
        
        if split_point=="ALT1":
            self.split_point  = "alt1"
            self.n_basis      = 224
            self.max_n_basis  = 224
            self.num_layers   = 3
            self.num_tile_x   = 14
            self.num_tile_y   = 16
            
            
        elif split_point=="DN53":
            self.split_point  = "dn53"
            self.n_basis      = 448
            self.max_n_basis  = 448
            self.num_layers   = 3
            self.num_tile_x   = 28
            self.num_tile_y   = 16
        self.num_tiles = (self.num_tile_y, self.num_tile_x)

        # vcm enc, dec params
        if self.n_basis > 0 or self.mode == "train":
            self.n_features = 7168 if self.split_point=="dn53" else 3584
            self.max_n_basis = 448 if self.split_point=="dn53" else 224
            self.num_chs    = [256, 512, 1024] if self.split_point=="dn53" else [128, 256, 512]

            self.trained_path = os.path.join(path, f"ft_restoration/mlt_track_{self.split_point}_{self.max_n_basis}x{self.n_features}_cfp.pt")
            sys.path.insert(0, os.path.join(path))
            print(self.trained_path)
            if self.mode=="train":
                self.ipca = IPCA(self.max_n_basis, self.n_features)
            else:
                self.ipca = torch.load(self.trained_path, map_location='cpu')

            self.ipca.to(device)
            self.ipca.prepare(self.n_basis)
            self.down = [ torch.nn.PixelUnshuffle(4), torch.nn.PixelUnshuffle(2), Dummy()]
            self.up   = [ torch.nn.PixelShuffle(4)  , torch.nn.PixelShuffle(2)  , Dummy()]
                

    def decode(self, f_packed, ft_shapes, poc, file_prefix):
        self.tile_size = (ft_shapes[2][1],ft_shapes[2][2])

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
        
        #removed Energy Compensation Scaling
        x = sorted(x, key=lambda x: x.shape[-1], reverse=False)
        
        return x
    
@register_ft_restoration("m65704-t")
class prebm_invpca_t(base):
    """
    Pre-defined basis&mean ( w/o Energy Compensation Scaling)
    m650705, Hannover, DE - October 2023
    Current implementation supports only for interfacing with R-CNN at FPN split points
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__(tag, device, **kwargs)
        split_coding_info = self.get_split_coding_info(
            tag, kwargs["split_point"], "dec"
        )
        self.poc = 0
        spif = split_coding_info["split_if"]
        self.spif = spif
        split_point = kwargs["split_point"]
        print("PreBM decoder is loaded ...")
        self.mlt = MLT(device="cpu",split_point=split_point)


    @torch.no_grad()
    def forward(self, ftensors, ft_shapes: List, file_prefix):
        d = self.to_device(ftensors)
        out = self.mlt.decode(d, ft_shapes, self.poc, file_prefix)
        self.poc=self.poc+1
        output_chs = [v[0] for v in ft_shapes]
        output_shapes = ft_shapes
        if output_chs != list(self.spif.values()):
            assert output_chs[::-1] == list(self.spif.values())
            output_shapes = ft_shapes[::-1]

        return self.sort_by_shape(out, output_shapes)
