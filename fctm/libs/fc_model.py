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
Top-level feature coding model class building a feature coder
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

import fctm.inner_codec
from fctm.libs.libdec import decode_ftensors
from fctm.libs.libenc import encode_ftensors
from fctm.utils.tools import DictConfig2Dict
from fctm.version import __version__

from .libcommon.common import CODING_MODULES, conversion
from .regitsters import FT_REDUCTION, FT_RESTORATION, INNER_CODECS

coding_behaviours_list = ["enc", "dec", "all"]


def build_feature_coding_model(
    coding_behaviour: str, tools: DictConfig, device: str, verbosity: int = 0
):
    assert coding_behaviour in coding_behaviours_list
    return feature_coding_model(
        coding_behaviour, DictConfig2Dict(tools), device, verbosity
    )


class feature_coding_model(nn.Module):
    def __init__(self, coding_behaviour: str, tools: Dict, device: str, verbosity: int):
        super().__init__()

        assert (
            coding_behaviour in coding_behaviours_list
        ), f"Unknown behaviour of coding, got {coding_behaviour} but should be {coding_behaviours_list}"

        self.device = torch.device(device)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._is_enc_cfg_printed = False

        self.verbosity = verbosity
        logging_level = logging.WARN
        if self.verbosity == 1:
            logging_level = logging.INFO
        if self.verbosity >= 2:
            logging_level = logging.DEBUG
        self.logger.setLevel(logging_level)

        self.coding_behaviour = coding_behaviour
        self.tools = tools

        self.setup_tools(tools, device)

        if self.coding_behaviour == "enc":
            self._sanity_check_for_configuration()

        self.reset()

    def __del__(self):
        self.close_bitstream_file()

    def reset(self):
        self._bitstream_fd = None
        return

    def open_bitstream_file(self, path, mode="rb"):
        self._bitstream_fd = open(path, mode)
        return self._bitstream_fd

    def close_bitstream_file(self):
        if self._bitstream_fd is not None:
            self._bitstream_fd.flush()
            self._bitstream_fd.close()
            self._bitstream_fd = None

    def _sanity_check_for_configuration(self):
        print("")

    def _print_enc_cfg(self, enc_cfg: Dict, lvl: int = 0):
        log_str = ""
        if lvl == 0 and self._is_enc_cfg_printed is True:
            return

        for key, val in enc_cfg.items():
            if isinstance(val, Dict):
                log_str += f"\n {' '*lvl}{'-' * lvl} {key} <"
                log_str += self._print_enc_cfg(val, (lvl + 1))
            else:
                sp = f"<{35-(lvl<<1)}s"
                log_str += f"\n {' '*lvl}{'-' * lvl} {str(key):{sp}} : {val}"

        if lvl == 0:
            intro = f"{'='*10} Coding configurations {'='*10}"
            endline = f"{'='*len(intro)}"
            log_str = f"\n {intro}" + log_str + f"\n {endline}" + "\n\n"
            self.logger.info(log_str)

            self._is_enc_cfg_printed = True

        return log_str

    def print_model_specs_and_configs(self):
        ops = {
            coding_behaviours_list[0]: "Encoder Only",
            coding_behaviours_list[1]: "Decoder Only",
            coding_behaviours_list[2]: "Encoder & Decoder",
        }

        log_str = ""
        log_str += f"\n\n {'='*80}"
        log_str += f"\n   << Feature Coding for Machines Software Test Model Version: {__version__} >>"
        log_str += f"\n\n    Coding behaviour : {ops[self.coding_behaviour]}"
        log_str += f"\n\n    -- List of selected modules in the coding pipeline"

        if not self.coding_behaviour == "dec":
            log_str += f"\n     [ FCM Encoder Pipeline]"
            for key, item in self.encoding_modules.items():
                log_str += f"\n       > {key:<27s}: {item.name}"

        if not self.coding_behaviour == "enc":
            log_str += f"\n     [ FCM Decoder Pipeline ]"
            for key, item in self.decoding_modules.items():
                log_str += f"\n       > {key:<27s}: {item.name}"

        log_str += f"\n {'='*80}\n"
        self.logger.info(log_str)

    def setup_tools(self, tools: Dict, device):
        _tag, _kwargs = self._retreive_tools(tools[str(CODING_MODULES.FT_RDCT)])
        self.ft_reduction = FT_REDUCTION[_tag](_tag, device, **_kwargs)

        _tag, _kwargs = self._retreive_tools(tools[str(CODING_MODULES.FT_RSTR)])
        self.ft_restoration = FT_RESTORATION[_tag](_tag, device, **_kwargs)

        _kwargs = tools[str(CODING_MODULES.CNVRS)]
        self.conversion = conversion(**_kwargs)

        _kwargs = tools[str(CODING_MODULES.INNER_CDC)]
        _kwargs["verbosity"] = self.verbosity
        self.inner_codec = INNER_CODECS[_kwargs["type"]](device, **_kwargs)

        self.encoding_modules = {
            CODING_MODULES.FT_RDCT: self.ft_reduction,
            CODING_MODULES.CNVRS: self.conversion,
            CODING_MODULES.INNER_CDC: self.inner_codec,
        }

        self.decoding_modules = {
            CODING_MODULES.INNER_CDC: self.inner_codec,
            CODING_MODULES.CNVRS: self.conversion,
            CODING_MODULES.FT_RSTR: self.ft_restoration,
        }

    @staticmethod
    def _retreive_tools(configs: Dict):
        selected = []
        for tool in configs:
            if configs[tool]["enabled"] is True:
                selected.append(tool)

        assert (
            len(selected) <= 1
        ), f"Too many tools are selected, {selected}, it should be one of them"

        if len(selected) == 1:
            return selected[0], configs[selected[0]]

        return "bypass", {}

    def encode(
        self,
        x: Dict,
        codec_output_dir,
        bitstream_name,
        file_prefix: str = "",
    ):
        self.reset()
        self.encoding_modules[CODING_MODULES.INNER_CDC].reset()

        self._print_enc_cfg(self.tools)

        if file_prefix == "":
            file_prefix = f"{codec_output_dir}/{bitstream_name}"
        else:
            file_prefix = f"{codec_output_dir}/{bitstream_name}-{file_prefix}"
        bitstream_path = f"{file_prefix}.bin"

        bitstream_fd = self.open_bitstream_file(bitstream_path, "wb")

        bytes_per_frame = encode_ftensors(
            bitstream_fd, x, self.encoding_modules, file_prefix
        )

        self.close_bitstream_file()

        return {
            "bytes": bytes_per_frame,
            "bitstream": bitstream_path,
        }

    def decode(
        self,
        input_path: str = None,
        codec_output_dir: str = "",
        file_prefix: str = "",
    ):
        self.reset()
        self.decoding_modules[CODING_MODULES.INNER_CDC].reset()

        output_file_prefix = Path(input_path).stem
        file_prefix = f"{codec_output_dir}/{output_file_prefix}"

        bitstream_fd = self.open_bitstream_file(input_path, "rb")

        output = decode_ftensors(bitstream_fd, self.decoding_modules, file_prefix)

        self.close_bitstream_file()

        return output
