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
Miscellenous functions
"""

from __future__ import annotations

import logging
import multiprocessing
import resource
import time
from pathlib import Path
from typing import Any, Dict, Union, cast

from omegaconf import DictConfig, OmegaConf


def DictConfig2Dict(configs: DictConfig, del_items=[]):
    _kwargs = OmegaConf.to_container(configs, resolve=True)

    for item in del_items:
        del _kwargs[item]

    return cast(Dict[str, Any], _kwargs)


def PathSetup(dir: Path) -> Path:
    path = Path(dir)

    if not path.is_dir():
        logging.info(f"creating output folder: {path}")
        path.mkdir(parents=True, exist_ok=True)

    return path


def get_filesize(filepath: Union[Path, str]) -> int:
    return Path(filepath).stat().st_size


def time_measure():
    return time.perf_counter()


def prevent_core_dump():
    # set no core dump at all
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def get_max_num_cpus():
    return multiprocessing.cpu_count()
