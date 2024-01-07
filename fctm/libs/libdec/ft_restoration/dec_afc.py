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


class AFC_decoder(nn.Module):
    def __init__(self, splits: List):
        super().__init__()

        self.ssfc = ConvBlock(64, splits[2], 1, 1, 0, affine=False)

        self.dec_toP2 = nn.Sequential(
            nn.Conv2d(splits[2], splits[1] * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(splits[1], affine=False),
        )
        self.dec_toP1 = nn.Sequential(
            nn.Conv2d(splits[2], splits[0] * 16, kernel_size=1, padding=0),
            nn.PixelShuffle(4),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(splits[0], affine=False),
        )

        self.dec_toC3 = Output_transform(
            bridge=ConvBlock(splits[2], splits[2], 1, 1, 0, affine=False),
            skip=nn.MaxPool2d(3, stride=1, padding=1),
        )
        self.out_p2 = Output_transform(
            bridge=ConvBlock(splits[1], splits[1], 3, 1, 1, affine=False),
            skip=nn.MaxPool2d(3, stride=1, padding=1),
        )
        self.out_p1 = Output_transform(
            bridge=ConvBlock(splits[0], splits[0], 3, 1, 1, affine=False),
            skip=nn.MaxPool2d(3, stride=1, padding=1),
        )

    def forward(self, feature_dec):
        feature_dec = self.ssfc(feature_dec)

        c3 = self.dec_toC3(feature_dec)
        p2 = self.out_p2(self.dec_toP2(feature_dec))
        p1 = self.out_p1(self.dec_toP1(feature_dec))

        return [p1, p2, c3]


@register_ft_restoration("m65181")
class dec_afc(base):
    """
    Asymmetrical Feature Coding
    m65181, Hannover, DE - October 2023
    Current implementation supports only for interfacing with JDE-YOLOv3 at two different split points
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__(tag, device, **kwargs)

        split_coding_info = self.get_split_coding_info(
            tag, kwargs["split_point"], "dec"
        )
        spif = split_coding_info["split_if"]
        self.spif = spif

        self.model_dec = AFC_decoder(list(spif.values())).to(device)
        state_dict = load_state_dict_from_server(split_coding_info["weight"], device)
        self.model_dec.load_state_dict(state_dict, strict=True)
        self.model_dec.eval()

    @torch.no_grad()
    def forward(self, ftensors, ft_shapes: List):
        d = self.to_device(ftensors)
        out = self.model_dec(d)

        output_chs = [v[0] for v in ft_shapes]
        output_shapes = ft_shapes
        if output_chs != list(self.spif.values()):
            assert output_chs[::-1] == list(self.spif.values())
            output_shapes = ft_shapes[::-1]

        return self.sort_by_shape(out, output_shapes)
