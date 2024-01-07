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
An entrace point of encoding for Feature Coding Software Test Model (FCTM).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from fctm.libs import build_feature_coding_model
from fctm.misc.env import get_env
from fctm.misc.git_configs import write_outputs
from fctm.utils.tools import PathSetup
from fctm.utils.utils import load_input_data

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../../cfgs").resolve())


def setup(conf: DictConfig) -> dict[str, Any]:
    conf.env = get_env(conf)

    codec = build_feature_coding_model(
        "enc", conf.codec.tools, conf.codec.device, conf.codec.verbosity
    )
    codec.print_model_specs_and_configs()

    input_path, kwargs = parsing_enc_configs(conf.codec)

    write_outputs(conf)

    return codec, input_path, kwargs


def parsing_enc_configs(configs: DictConfig):
    input_feature_tensors = Path(configs.input_feature_tensors)
    assert (
        input_feature_tensors.is_file()
    ), f"Can't read a file, check your file {input_feature_tensors}"

    PathSetup(Path(configs.output_dir))

    return str(input_feature_tensors.resolve()), {
        "codec_output_dir": str(configs.output_dir),
        "bitstream_name": str(configs.bitstream_name),
        "file_prefix": str(input_feature_tensors.stem),
    }


@hydra.main(version_base=None, config_path=str(config_path))
def main(conf: DictConfig):
    codec, input_path, kwargs = setup(conf)

    d = load_input_data(conf, input_path, map_location="cpu")

    out = codec.encode(d, **kwargs)

    print(
        f":: Encoding is done. \
          \nOutput bitstream, {Path(out['bitstream']).name}, \
          \n is available at \"{os.getcwd()}/{Path(out['bitstream']).parent}\" \
          \n"
    )

    return out


if __name__ == "__main__":
    main()
