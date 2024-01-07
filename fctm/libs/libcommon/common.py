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

import math
from enum import Enum
from typing import Dict, Tuple, Union

import torch.nn as nn
from torch import Tensor


class CODING_MODULES(Enum):
    FT_RDCT = "Feature_Reduction"
    FT_RSTR = "Feature_Restoration"
    CNVRS = "Conversion"
    INNER_CDC = "Inner_Codec"

    def __str__(self):
        return f"{self.value.lower()}"

    def __format__(self, spec):
        return f"{self.value}"


class conversion(nn.Module):
    """
    Default conversion class supporting basic normalization,
    uniform scalar quantization, and frame packing
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.bypass = kwargs["bypass"]
        self.n_bit = kwargs["n_bit"]
        self.ch_in_width = kwargs["packing"]["channels_in_width"]
        self.ch_in_height = kwargs["packing"]["channels_in_height"]

        if self.bypass is False:
            if self.ch_in_width == -1 or self.ch_in_height == -1:
                self._packing = "unknown"
            else:
                self._packing = f"{self.ch_in_height}x{self.ch_in_width}(chs_in_height x chs_in_width)"

    @property
    def name(self):
        if self.bypass:
            return "bypass"

        return f"{self.n_bit}_bit_quant+frame_packing_with_{self._packing}"

    def forward(self, x: Union[Tensor, Dict]) -> Union[Tensor, Dict]:
        if self.bypass:
            return x

        assert (
            x.dim() == 3
        ), f"3 dimension of the input data is expected, but got {x.shape}"

        assert isinstance(x, Tensor)

        C, H, W = x.shape
        if self._packing == "unknown":
            fh, fw = compute_frame_resolution(C, H, W)
        else:
            assert (
                self.ch_in_height * self.ch_in_width == C
            ), f"Channel size mismatch, {C} must be equal to ch_in_height x ch_in_width, but got {self.ch_in_height}x{self.ch_in_width}"
            fh = H * self.ch_in_height
            fw = W * self.ch_in_width

        out = tensor_to_tiled(x, (fh, fw))

        minv = out.min()
        maxv = out.max()

        out_frame, _ = min_max_normalization(out, minv, maxv, bitdepth=self.n_bit)

        return {
            "frm": out_frame,
            "chSize": (H, W),
            "minv": minv,
            "maxv": maxv,
            "n_bit": self.n_bit,
        }

    def inverse(
        self, x: Union[Tensor, Dict], minv=-1, maxv=-1, chSize=None, n_bit: int = 10
    ) -> Union[Tensor, Dict]:
        if self.bypass:
            assert (
                "ftensor" in x
            ), "There is keyword 'ftensor' in given input dictionary"
            return x

        assert (
            n_bit == self.n_bit
        ), f"Mismatch found between given n_bit={n_bit} at the decoder and {self.n_bit} used at the encoder"

        assert chSize is not None and isinstance(x, Tensor)

        ofrm = min_max_inv_normalization(x, minv, maxv, bitdepth=n_bit)

        ftensor = tiled_to_tensor(ofrm, chSize)

        return ftensor


def compute_frame_resolution(num_channels, channel_height, channel_width):
    r"""Codes below referred to `"CompressAI-Vision"
    <https://github.com/InterDigitalInc/CompressAI-Vision>`,
    which is licensed under BSD-3-Clause-Clear.

    Full lincense statement can be found at
    <https://github.com/InterDigitalInc/CompressAI-Vision/blob/main/LICENSE>
    """

    short_edge = int(math.sqrt(num_channels))

    while (num_channels % short_edge) != 0:
        short_edge -= 1

    long_edge = num_channels // short_edge

    assert (short_edge * long_edge) == num_channels

    # try to make the shape close to a square
    if channel_height > channel_width:
        height = short_edge * channel_height
        width = long_edge * channel_width
    else:
        width = short_edge * channel_width
        height = long_edge * channel_height

    return (height, width)


def tensor_to_tiled(x: Tensor, tiled_frame_resolution):
    r"""Codes below referred to `"CompressAI-Vision"
    <https://github.com/InterDigitalInc/CompressAI-Vision>`,
    which is licensed under BSD-3-Clause-Clear.

    Full lincense statement can be found at
    <https://github.com/InterDigitalInc/CompressAI-Vision/blob/main/LICENSE>
    """

    assert x.dim() == 3 and isinstance(x, Tensor)
    _, H, W = x.size()

    num_channels_in_height = tiled_frame_resolution[0] // H
    num_channels_in_width = tiled_frame_resolution[1] // W

    x = x[None, :, :, :]
    A = x.reshape(num_channels_in_height, num_channels_in_width, H, W)
    B = A.swapaxes(1, 2)
    tiled = B.reshape(tiled_frame_resolution[0], tiled_frame_resolution[1])

    return tiled


def tiled_to_tensor(x: Tensor, channel_resolution):
    r"""Codes below referred to `"CompressAI-Vision"
    <https://github.com/InterDigitalInc/CompressAI-Vision>`,
    which is licensed under BSD-3-Clause-Clear.

    Full lincense statement can be found at
    <https://github.com/InterDigitalInc/CompressAI-Vision/blob/main/LICENSE>
    """

    assert x.dim() == 2 and isinstance(x, Tensor)
    frmH, frmW = x.size()

    num_channels_in_height = frmH // channel_resolution[0]
    num_channels_in_width = frmW // channel_resolution[1]
    total_num_channels = int(num_channels_in_height * num_channels_in_width)

    A = x.reshape(
        num_channels_in_height,
        channel_resolution[0],
        num_channels_in_width,
        channel_resolution[1],
    )
    B = A.swapaxes(1, 2)
    feature_tensor = B.reshape(
        1, total_num_channels, channel_resolution[0], channel_resolution[1]
    )

    return feature_tensor


def min_max_normalization(x, min: float, max: float, bitdepth: int = 10):
    r"""Codes below referred to `"CompressAI-Vision"
    <https://github.com/InterDigitalInc/CompressAI-Vision>`,
    which is licensed under BSD-3-Clause-Clear.

    Full lincense statement can be found at
    <https://github.com/InterDigitalInc/CompressAI-Vision/blob/main/LICENSE>
    """
    max_num_bins = (2**bitdepth) - 1
    out = ((x - min) / (max - min)).clamp_(0, 1)
    mid_level = -min / (max - min)
    return (out * max_num_bins).floor(), int(mid_level * max_num_bins + 0.5)


def min_max_inv_normalization(x, min: float, max: float, bitdepth: int = 10):
    r"""Codes below referred to `"CompressAI-Vision"
    <https://github.com/InterDigitalInc/CompressAI-Vision>`,
    which is licensed under BSD-3-Clause-Clear.

    Full lincense statement can be found at
    <https://github.com/InterDigitalInc/CompressAI-Vision/blob/main/LICENSE>
    """
    out = x / ((2**bitdepth) - 1)
    out = (out * (max - min)) + min
    return out
