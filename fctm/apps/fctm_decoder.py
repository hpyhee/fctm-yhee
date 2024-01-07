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
An entrace point of decoding for Feature Coding Software Test Model (FCTM).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, open_dict

from fctm.libs import build_feature_coding_model
from fctm.misc.env import get_env
from fctm.misc.git_configs import write_outputs
from fctm.utils.tools import PathSetup
from fctm.utils.utils import write_output_data

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../../cfgs").resolve())


def setup(conf: DictConfig) -> dict[str, Any]:
    conf.env = get_env(conf)

    with open_dict(conf):
        conf.codec.tools.inner_codec.inspection_mode = conf.inspection_mode

    codec = build_feature_coding_model(
        "dec",
        conf.codec.tools,
        conf.misc.device,
        conf.codec.verbosity,
    )
    codec.print_model_specs_and_configs()

    input_path, kwargs = parsing_dec_configs(conf.codec)

    write_outputs(conf)

    return codec, input_path, kwargs


def parsing_dec_configs(configs: DictConfig):
    input_bitstream = Path(configs.input_feature_tensors)

    assert (
        input_bitstream.is_file()
    ), f"Can't read a bitstream, check your bitstream file {input_bitstream}"

    PathSetup(Path(configs.output_dir))

    return str(input_bitstream.resolve()), {
        "codec_output_dir": str(configs.output_dir),
        "file_prefix": str(configs.output_name),
    }


@hydra.main(version_base=None, config_path=str(config_path))
def main(conf: DictConfig):
    codec, input_path, kwargs = setup(conf)

    out = codec.decode(input_path, **kwargs)

    res = write_output_data(conf, out)

    print(
        f':: Decoding is done. \
          \nOutput data, {Path(res).name}, \
          \n is available at "{os.getcwd()}/{Path(res).parent}" \
          \n'
    )

    return out


if __name__ == "__main__":
    main()
