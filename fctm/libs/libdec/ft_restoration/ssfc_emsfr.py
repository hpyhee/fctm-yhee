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

import fvcore.nn.weight_init as weight_init  # fvcore==0.1.5.post20210812
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fctm.libs.regitsters import register_ft_restoration
from fctm.utils.utils import load_state_dict_from_server

from ...libcommon.default import base


class enhanced_MSFR(nn.Module):
    def __init__(self, in_channels_per_feature):
        super(enhanced_MSFR, self).__init__()
        MSFR_convs, MSFR_convs2 = [], []
        for idx, in_channels in enumerate(in_channels_per_feature):
            MSFR_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            MSFR_conv2 = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            weight_init.c2_xavier_fill(MSFR_conv)
            weight_init.c2_xavier_fill(MSFR_conv2)
            self.add_module("MSFR_conv_{}".format(idx), MSFR_conv)
            self.add_module("MSFR_conv2_{}".format(idx), MSFR_conv2)
            MSFR_convs.append(MSFR_conv)
            MSFR_convs2.append(MSFR_conv2)
        self.MSFR_convs = MSFR_convs[::-1]
        self.MSFR_convs2 = MSFR_convs2[::-1]

        downsamples = []
        for idx, in_channels in enumerate(in_channels_per_feature):
            stride = 2
            downsample = nn.AvgPool2d(stride, stride=stride)
            self.add_module("downsample_{}".format(idx), downsample)
            downsamples.append(downsample)
        self.dowmsamples = downsamples
        self.num_of_feautre_map = len(in_channels_per_feature)
        self.make_p6 = nn.AvgPool2d(2, 2)

    def forward(self, x):  # top-down MSFR
        f_ssfc = x
        results = []
        for idx, (MSFR_conv, MSFR_conv2, downsmaple) in enumerate(
            zip(self.MSFR_convs, self.MSFR_convs2, self.dowmsamples)
        ):
            scale_factor = 2**idx
            msfr_features = F.interpolate(
                f_ssfc, scale_factor=scale_factor, mode="nearest"
            )
            if idx == 0:
                prev_features = MSFR_conv(msfr_features)
            else:
                result = F.interpolate(results[0], scale_factor=2, mode="nearest")
                prev_features = MSFR_conv2(MSFR_conv(result) + msfr_features)

            results.insert(0, prev_features)

        return list(reversed(results))


class SSFC_decoder(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(SSFC_decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channel, input_channel, kernel_size=1),
            nn.BatchNorm2d(input_channel),
            nn.PReLU(),
        )
        weight_init.c2_xavier_fill(self.decoder[0])

    def forward(self, x):
        return self.decoder(x)


class eMSFC_decoder(nn.Module):
    def __init__(self, splits, c_prime):
        super(eMSFC_decoder, self).__init__()
        self.SSFC_decoder = SSFC_decoder(min(splits), c_prime)
        self.MSFR = enhanced_MSFR(splits)

    def forward(self, x):
        x = self.SSFC_decoder(x)
        x = self.MSFR(x)
        return x


@register_ft_restoration("m65705")
class ssfc_emsfr(base):
    """
    Single-stream Feature Codec + Enhanced Multi-scale Feature Reconstruction
    m650705, Hannover, DE - October 2023
    Current implementation supports only for interfacing with R-CNN at FPN split points
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__(tag, device, **kwargs)

        split_coding_info = self.get_split_coding_info(tag, kwargs["task"], "dec")

        spif = split_coding_info["split_if"]
        self.eMSFC_decoder = eMSFC_decoder(spif.values(), c_prime=64).to(device)

        state_dict = load_state_dict_from_server(split_coding_info["weight"], device)
        self.eMSFC_decoder.load_state_dict(state_dict, strict=True)
        self.eMSFC_decoder.eval()

    @torch.no_grad()
    def forward(self, ftensors: Tensor, ft_shapes):
        d = self.to_device(ftensors)
        x_data = self.eMSFC_decoder(d)
        return self.sort_by_shape(x_data, ft_shapes)
