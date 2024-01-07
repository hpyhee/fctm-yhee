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

from ...libcommon.default import base


class SSFC_encoder(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(SSFC_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=1),
            nn.BatchNorm2d(hidden_channel),
            nn.PReLU(),
        )

        weight_init.c2_xavier_fill(self.encoder[0])

    def forward(self, x):
        return self.encoder(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MSFF(nn.Module):
    def __init__(self, planes, num_of_feature_map):
        super(MSFF, self).__init__()
        assert planes, num_of_feature_map
        self.se = SELayer(planes * num_of_feature_map, reduction=64)
        # print(planes, num_of_feature_map, planes * num_of_feature_map)
        self.conv1 = nn.Conv2d(
            planes * num_of_feature_map,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        weight_init.c2_xavier_fill(self.conv1)
        aligns = []
        for idx in range(num_of_feature_map):
            stride = 1 if idx == 0 else 2 << (idx - 1)
            align = nn.AvgPool2d(stride, stride=stride)
            self.add_module("align_{}".format(stride), align)
            aligns.append(align)

        self.aligns = aligns

    def forward(self, x_lst):
        for idx, (p_feature, align) in enumerate(zip(x_lst, self.aligns)):
            x_lst[idx] = align(x_lst[idx])
        x = torch.cat((x_lst[:]), 1)
        x = self.conv1(self.se(x))

        return x


class eMSFC_encoder(nn.Module):
    def __init__(self, splits, c_prime):
        super(eMSFC_encoder, self).__init__()
        self.MSFF = MSFF(min(splits), len(splits))
        self.SSFC_encoder = SSFC_encoder(min(splits), c_prime)

    def forward(self, x):
        x = self.MSFF(x)
        x = self.SSFC_encoder(x)
        return x


@register_ft_reduction("m65705")
class msff_ssfc(base):
    """
    Multi-scale Feature Fusion + Single-stream Feature Codec
    m650705, Hannover, DE - October 2023
    Current implementation supports only for interfacing with R-CNN at FPN split points
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__(tag, device, **kwargs)

        split_coding_info = self.get_split_coding_info(tag, kwargs["task"], "enc")

        spif = split_coding_info["split_if"]
        self.eMSFC_encoder = eMSFC_encoder(spif.values(), c_prime=64).to(device)

        state_dict = load_state_dict_from_server(split_coding_info["weight"], device)
        self.eMSFC_encoder.load_state_dict(state_dict, strict=True)
        self.eMSFC_encoder.eval()

    @torch.no_grad()
    def forward(self, ftensors):
        data = [self.to_device(v.unsqueeze(0)) for v in ftensors.values()][::-1]
        data = self.eMSFC_encoder(data)
        return data.squeeze()
