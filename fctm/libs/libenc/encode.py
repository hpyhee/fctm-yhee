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
Feature compression encoder
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch.nn as nn

from fctm.libs.libcommon.readwrite import *
from fctm.utils.utils import iterate_list_of_ftensors

from ..libcommon.common import CODING_MODULES


# Adhoc: write a feature tensors parameter set (FTPS, temporarily named) to make bitstream self-decodable
# [TODO: Later formal name and class would be needed and this function would be replaced by that]
def write_feature_tensors_parameter_set(fd: Any, x: Dict):
    assert set(["input_size", "org_input_size", "data"]).issubset(x)

    byte_cnts = 0

    org_input_height, org_input_width = x["org_input_size"].values()
    input_height, input_width = x["input_size"][0]

    shapes_of_ftensor = []
    for item in x["data"].values():
        assert item.dim() == 4
        shapes_of_ftensor.append(item.shape[1:])

    # write original input resolution
    byte_cnts += write_uints(fd, (org_input_height, org_input_width))

    # write input resolution
    byte_cnts += write_uints(fd, (input_height, input_width))

    # write number of feature sets (1 byte)
    byte_cnts += write_uchars(fd, (len(shapes_of_ftensor),))

    # write feature tensor dims (4 bytes each - temp): nb_channels, H, W
    for dshape in shapes_of_ftensor:
        byte_cnts += write_uints(fd, dshape)

    return byte_cnts


def encode_ftensors(
    bitstream_fd: Any,
    x: Dict,
    cmd: Dict[CODING_MODULES, Callable[..., nn.Module]],
    file_prefix: str,
):
    bytes_per_frame = []

    # check the batch size
    nbframes_per_layer = [layer_data.size()[0] for _, layer_data in x["data"].items()]
    assert all(n == nbframes_per_layer[0] for n in nbframes_per_layer)
    nbframes = nbframes_per_layer[0]

    for e, ftensors in iterate_list_of_ftensors(x["data"]):
        hls_bytes = 0
        # Adhoc: write a feature tensors parameter set (FPS, temporarily named) to make bitstream self-decodable
        # print("e: ", e )
        # print("file_prefix: str: ", file_prefix )
        if e == 0:
            hls_bytes = write_feature_tensors_parameter_set(bitstream_fd, x)

        # feature reduction
        # out = cmd[CODING_MODULES.FT_RDCT](ftensors)
        out = cmd[CODING_MODULES.FT_RDCT](ftensors, file_prefix)

        # conversion
        out = cmd[CODING_MODULES.CNVRS](out)

        # inner coding
        out = cmd[CODING_MODULES.INNER_CDC].encode(
            out, file_prefix, (e + 1) == nbframes
        )

        bytes_per_frame.append(hls_bytes)

    assert set(["bytes", "bitstream"]).issubset(out)

    bitstream_fd.write(out["bitstream"])

    # update bytes (Adhoc)
    for e, main_bytes in enumerate(out["bytes"]):
        bytes_per_frame[e] += main_bytes

    return bytes_per_frame
