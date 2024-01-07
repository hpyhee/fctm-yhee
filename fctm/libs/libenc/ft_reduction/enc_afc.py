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


import torch
import torch.nn as nn
import torch.nn.functional as F

from fctm.libs.regitsters import register_ft_reduction
from fctm.utils.utils import load_state_dict_from_server

from ...libcommon.default import base


class ConvBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBlock, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class ECA_block(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Output_transform(nn.Module):
    def __init__(self, bridge=None, skip=None):
        super().__init__()

        self.bridge = bridge
        self.skip = skip

    def forward(self, x):
        return self.bridge(x) + self.skip(x)


class AFC_encoder(nn.Module):
    def __init__(self, splits=None):
        super().__init__()

        self.c3_align = ConvBlock(splits[2], splits[2], 1, 1, 0, affine=False)
        self.p2_align = ConvBlock(splits[1], splits[1], 3, 2, 1, affine=False)
        self.p1_align = nn.Sequential(
            ConvBlock(splits[0], splits[0], 3, 2, 1, affine=False),
            ConvBlock(splits[0], splits[0], 3, 2, 1, affine=False),
        )

        self.attn = ECA_block(sum(splits))  # (512+256+128) * 2
        self.ch_down = ConvBlock(sum(splits), 64, 1, 1, 0, affine=False)

    def forward(self, D):
        c3 = self.c3_align(D[2])
        p2 = self.p2_align(D[1])
        p2 += F.interpolate(D[1], scale_factor=0.5, mode="bilinear")
        p1 = self.p1_align(D[0])
        p1 += F.interpolate(D[0], scale_factor=0.25, mode="bilinear")
        feature_enc = self.attn(torch.cat([c3, p2, p1], 1))
        feature_enc = self.ch_down(feature_enc)

        return feature_enc


@register_ft_reduction("m65181")
class enc_afc(base):
    """
    Asymmetrical Feature Coding
    m65181, Hannover, DE - October 2023
    Current implementation supports only for interfacing with JDE-YOLOv3 at two different split points
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__(tag, device, **kwargs)

        split_coding_info = self.get_split_coding_info(
            tag, kwargs["split_point"], "enc"
        )
        spif = split_coding_info["split_if"]

        self.chs_in_order = list(spif.values())

        self.model_enc = AFC_encoder(splits=self.chs_in_order).to(device)
        state_dict = load_state_dict_from_server(split_coding_info["weight"], device)
        self.model_enc.load_state_dict(state_dict, strict=True)
        self.model_enc.eval()

    @torch.no_grad()
    def forward(self, ftensors):
        input_chs = [v.shape[0] for v in ftensors.values()]
        data = [self.to_device(v.unsqueeze(0)) for v in ftensors.values()]

        if input_chs != self.chs_in_order:
            data = data[::-1]
            assert input_chs[::-1] == self.chs_in_order

        out = self.model_enc(data)
        return out.squeeze()
