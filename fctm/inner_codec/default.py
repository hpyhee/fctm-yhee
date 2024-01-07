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


from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

from fctm.libs.libcommon.readwrite import *
from fctm.libs.regitsters import register_inner_codec
from fctm.utils.utils import generate_frame


@register_inner_codec("bypass")
class inner_codec(nn.Module):
    """Common class for inner codec"""

    def __init__(
        self,
        device,
        **kwargs,
    ):
        super().__init__()

        self.device = device
        self.inner_codec_type = kwargs["type"]
        self.enc_cfgs = kwargs["enc_configs"]

        self.stash_outputs = kwargs["stash_outputs"]

        self.inspection_mode = False
        if "inspection_mode" in kwargs:
            self.inspection_mode = kwargs["inspection_mode"]

        self.reset()

    @property
    def codec_type(self):
        return self.inner_codec_type

    @property
    def name(self):
        return self.inner_codec_type

    def reset(self):
        self._input_buffer = []
        self._min_max_buffer = []
        self._temp_io_buffer = BytesIO()

    def encode(
        self,
        x: Dict,
        file_prefix: str,
        last_input: bool,
    ) -> bool:
        # check items to register
        assert set(["frm", "chSize", "minv", "maxv", "n_bit"]).issubset(x)

        frmH, frmW = self.register_frame_to_input_buffer(x["frm"])
        self.register_minmax_to_buffer(x["minv"], x["maxv"])

        if last_input is False:
            return None

        nbframes = self.get_input_buffer_size()

        assert self.get_minmax_buffer_size() == nbframes

        copied = x.copy()
        del copied["minv"]
        del copied["maxv"]

        copied["frm"] = self.input_buffer
        copied["minv"] = [minv for minv, _ in self.minmax_buffer]
        copied["maxv"] = [maxv for _, maxv in self.minmax_buffer]
        self.copied_for_bypass = copied

        # adhoc method to make bitstream self-decodable
        _ = self.write_n_bit(x["n_bit"])
        _ = self.write_rft_chSize(x["chSize"])
        _ = self.write_packed_frame_size((frmH, frmW))
        _ = self.write_min_max_values()

        avg_bytes_per_frame = 9999999
        all_bytes_per_frame = [avg_bytes_per_frame] * nbframes

        return {
            "bytes": all_bytes_per_frame,
            "bitstream": self.get_io_buffer_contents(),
        }

    def decode(
        self,
        bitstream_fd: Any,
        file_prefix: str = "",
    ) -> Dict:
        # make sure nothing came from encoder
        self.reset()

        n_bit = self.read_n_bit(bitstream_fd)
        chH, chW = self.read_rft_chSize(bitstream_fd)
        frmH, frmW = self.read_packed_frame_size(bitstream_fd)
        _min_max_buffer = self.read_min_max_values(bitstream_fd)

        if self.inspection_mode is False:
            return self.copied_for_bypass

        nbframes = len(_min_max_buffer)

        rec_frames = []
        list_of_minv = []
        list_of_maxv = []
        for i in range(nbframes):
            rec_yuv = generate_frame((frmH, frmW), n_bit)
            minv, maxv = _min_max_buffer[i]

            rec_frames.append(rec_yuv)
            list_of_minv.append(minv)
            list_of_maxv.append(maxv)

        return {
            "frm": rec_frames,
            "chSize": (chH, chW),
            "minv": list_of_minv,
            "maxv": list_of_maxv,
            "n_bit": n_bit,
        }

    def register_frame_to_input_buffer(self, frm: Tensor):
        x = frm.cpu()
        self._input_buffer.append(x)

        return x.shape

    def register_minmax_to_buffer(self, minv, maxv):
        assert minv.dtype == maxv.dtype == torch.float32
        self._min_max_buffer.append((float(minv), float(maxv)))

    def get_input_buffer_size(self):
        return len(self._input_buffer)

    @property
    def minmax_buffer(self):
        return self._min_max_buffer

    def get_minmax_buffer_size(self):
        return len(self._min_max_buffer)

    def get_io_buffer_contents(self):
        return self._temp_io_buffer.getvalue()

    def write_n_bit(self, n_bit):
        # adhoc method, warning redundant information
        return write_uchars(self._temp_io_buffer, (n_bit,))

    def write_rft_chSize(self, chSize):
        # adhoc method
        return write_uints(self._temp_io_buffer, chSize)

    def write_packed_frame_size(self, frmSize):
        # adhoc method, warning redundant information
        return write_uints(self._temp_io_buffer, frmSize)

    def write_min_max_values(self):
        # adhoc method to make bitstream self-decodable
        byte_cnts = write_uints(self._temp_io_buffer, (self.get_minmax_buffer_size(),))
        for min_max in self._min_max_buffer:
            byte_cnts += write_float32(self._temp_io_buffer, min_max)

        return byte_cnts

    def read_n_bit(self, fd):
        # adhoc method, warning redundant information
        return read_uchars(fd, 1)[0]

    def read_rft_chSize(self, fd):
        # adhoc method,
        return read_uints(fd, 2)

    def read_packed_frame_size(self, fd):
        # adhoc method, warning redundant information
        return read_uints(fd, 2)

    def read_min_max_values(self, fd):
        # adhoc method to make bitstream self-decodable
        num_minmax_pairs = read_uints(fd, 1)[0]

        min_max_buffer = []
        for _ in range(num_minmax_pairs):
            min_max = read_float32(fd, 2)
            min_max_buffer.append(min_max)

        return min_max_buffer

    @property
    def input_buffer(self):
        return self._input_buffer

    def load_up_bitstream(self, path):
        with open(path, "rb") as fd:
            buf = BytesIO(fd.read())

        return buf.getvalue()
