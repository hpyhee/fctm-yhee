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

r"""
Feature compression decoder
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch.nn as nn
from torch import Tensor

from fctm.libs.libcommon.readwrite import *

from ..libcommon.common import CODING_MODULES


# Adhoc: read a feature tensors parameter set (FPS, temporarily named) to make bitstream self-decodable
# [TODO: Later formal name and class would be needed and this function would be replaced by that]
def read_feature_tensors_parameter_set(fd: Any):
    # read original input resolution
    org_input_height, org_input_width = read_uints(fd, 2)

    # read input resolution
    input_height, input_width = read_uints(fd, 2)

    # read number of feature sets (1 byte)
    num_of_ftensors = read_uchars(fd, 1)[0]

    # read feature tensor dims (4 bytes each - temp): nb_channels, H, W
    ft_shapes = []
    for _ in range(num_of_ftensors):
        C, H, W = read_uints(fd, 3)
        ft_shapes.append((C, H, W))

    output = {
        "org_input_size": {
            "height": org_input_height,
            "width": org_input_width,
        },
        "input_size": [(input_height, input_width)],
        "data": None,
    }

    return output, ft_shapes


def decode_ftensors(
    bitstream_fd: Any,
    cmd: Dict[CODING_MODULES, Callable[..., nn.Module]],
    file_prefix: str,
):
    # adhoc: read feature tensors parameter set (FPS)
    output, ft_shapes = read_feature_tensors_parameter_set(bitstream_fd)

    # temporary tag name
    # it should be replaced outside of decoder with correct name tag to be compatible with NN-Part2
    ftensor_tags = [i for i in range(len(ft_shapes))]
    recon_ftensors = dict(zip(ftensor_tags, [[] for _ in range(len(ftensor_tags))]))

    # inner coding
    out = cmd[CODING_MODULES.INNER_CDC].decode(bitstream_fd, file_prefix)

    assert set(["frm", "chSize", "minv", "maxv", "n_bit"]).issubset(out)

    chSize = out["chSize"]
    n_bits = out["n_bit"]

    for _, items in enumerate(zip(out["frm"], out["minv"], out["maxv"])):
        frm, minv, maxv = items
        # conversion
        out = cmd[CODING_MODULES.CNVRS].inverse(frm, minv, maxv, chSize, n_bits)

        # feature restoration
        # res_ftensor = cmd[CODING_MODULES.FT_RSTR](out, ft_shapes)
        res_ftensor = cmd[CODING_MODULES.FT_RSTR](out, ft_shapes, file_prefix)

        for tlist, ftensor in zip(recon_ftensors.values(), res_ftensor):
            assert isinstance(ftensor, Tensor)

            if ftensor.dim() == 4:
                ftensor = ftensor.squeeze(0)

            assert ftensor.dim() == 3

            tlist.append(ftensor)

    output["data"] = recon_ftensors

    return output
