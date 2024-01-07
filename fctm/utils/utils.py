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
Utility functions
"""

from __future__ import annotations

import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .tools import get_max_num_cpus, prevent_core_dump


def iterate_list_of_ftensors(data: Dict):
    list_of_features_sets = list(data.values())
    list_of_keys = list(data.keys())

    num_feature_sets = list_of_features_sets[0].size(0)

    if any(fs.size(0) != num_feature_sets for fs in list_of_features_sets):
        raise ValueError("All feature tensors must have the same number of batches")

    for e, current_feature_set in enumerate(zip(*list_of_features_sets)):
        yield e, dict(zip(list_of_keys, current_feature_set))


def load_state_dict_from_server(tag: str, device):
    root_url = "https://dspub.blob.core.windows.net/mpeg-fcm/weights"
    url = root_url + "/" + tag
    return load_state_dict_from_url(
        url, progress=True, check_hash=True, map_location=device
    )


def load_input_data(conf: DictConfig, input_path: Path, map_location: str = "cpu"):
    d = torch.load(input_path, map_location=map_location)

    out_d = d.copy()
    if conf.inspection_mode is True:
        assert "ftshapes" in d

        ftshapes = d["ftshapes"]
        del out_d["ftshapes"]

        torch.random.manual_seed(conf.misc.seed)
        for k, shape in zip(out_d["data"], ftshapes):
            C, H, W = shape
            ft = torch.rand(2, C, H, W)
            out_d["data"][k] = ft

    return out_d


def write_output_data(conf: DictConfig, output: Dict):
    output_file = str(conf.codec.output_dir) + "/" + str(conf.codec.output_name)

    if conf.inspection_mode is False:
        torch.save(output, output_file)
        return output_file

    d = output.copy()
    fts = d["data"]
    del d["data"]

    ftshapes = []
    for _, tensor in fts.items():
        if isinstance(tensor, Tensor):
            assert tensor.dim() == 4
            ftshapes.append(tensor.shape[1:])
        elif isinstance(tensor, List):
            assert tensor[0].dim() == 3
            ftshapes.append(tensor[0].shape)
        else:
            raise NotImplementedError

    ftensor_tags = [i for i in range(len(ftshapes))]
    d["data"] = dict(zip(ftensor_tags, [[] for _ in range(len(ftensor_tags))]))
    d["ftshapes"] = ftshapes

    torch.save(d, output_file)

    return output_file


def generate_frame(size: Tuple, n_bit: int):
    torch.random.manual_seed(1234)
    max_lvl = (2**n_bit) - 1
    frm = torch.rand(size).cpu()

    minv = frm.min()
    maxv = frm.max()

    out = ((frm - minv) / (maxv - minv)) * max_lvl

    return torch.round(out)


def run_cmdline(cmds: List[Any], logpath: Optional[Path] = None) -> None:
    def worker(cmd, id, logpath):
        print(f"--> job_id [{id:03d}] Running: {' '.join(cmd)}", file=sys.stderr)
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=prevent_core_dump,
        )

        if logpath is not None:
            plogpath = Path(str(logpath) + f".sub_p{id}")
            with plogpath.open("w") as f:
                for bline in p.stdout:
                    line = bline.decode()
                    f.write(line)
                f.flush()
            assert p.wait() == 0
        else:
            p.stdout.read()  # clear up

    with cf.ThreadPoolExecutor(get_max_num_cpus()) as exec:
        all_jobs = [
            exec.submit(worker, cmd, id, logpath) for id, cmd in enumerate(cmds)
        ]
        cf.wait(all_jobs)

    return


def inst_run_cmdline(cmdline: List[Any]) -> None:
    cmdline = list(map(str, cmdline))
    print(f"--> Running: {' '.join(cmdline)}", file=sys.stderr)
    out = subprocess.check_output(cmdline).decode()
    if out:
        print(out)

    return
