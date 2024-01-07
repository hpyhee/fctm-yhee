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

import errno
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from fctm.libs.regitsters import register_inner_codec
from fctm.utils.dataio import bitdepth_to_YUV400format, readwriteYUV
from fctm.utils.tools import get_filesize, time_measure
from fctm.utils.utils import inst_run_cmdline, run_cmdline

from .default import inner_codec

r"""Most codes below referred to `"CompressAI-Vision"
    <https://github.com/InterDigitalInc/CompressAI-Vision>`,
    which is licensed under BSD-3-Clause-Clear.

    Full lincense statement can be found at 
    <https://github.com/InterDigitalInc/CompressAI-Vision/blob/main/LICENSE>
"""


@register_inner_codec("vtm")
class VTM(inner_codec):
    """Encoder/Decoder class for VVC - VTM reference software"""

    def __init__(
        self,
        device,
        **kwargs,
    ):
        super().__init__(device, **kwargs)

        self.encoder_path = Path(kwargs["enc_exe"])
        self.decoder_path = Path(kwargs["dec_exe"])
        self.cfg_file = Path(self.enc_cfgs["cfg_file"])
        self.merger_path = Path(kwargs["merge_exe"])  # optional
        self.parallel_encoding = self.enc_cfgs["parallel_encoding"]  # parallel option
        self.hash_check = self.enc_cfgs["hash_check"]  # md5 hash check

        check_list_of_paths = [self.encoder_path, self.decoder_path, self.cfg_file]
        if self.parallel_encoding:  # miminum
            check_list_of_paths.append(self.merger_path)

        for file_path in check_list_of_paths:
            if not file_path.is_file():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), file_path
                )

        self.bitdepth = self.enc_cfgs["n_bit"]
        self.qp = self.enc_cfgs["qp"]
        self.intra_period = self.enc_cfgs["intra_period"]
        self.frame_rate = self.enc_cfgs["frame_rate"]

        self.yuvio = readwriteYUV(
            device="cpu", format=bitdepth_to_YUV400format[self.bitdepth]
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbosity = kwargs["verbosity"]
        logging_level = logging.WARN
        if self.verbosity == 1:
            logging_level = logging.INFO
        if self.verbosity >= 2:
            logging_level = logging.DEBUG

        self.logger.setLevel(logging_level)

        self.reset()

    def get_encode_cmd(
        self,
        inp_yuv_path: Path,
        qp: int,
        bitstream_path: str,
        width: int,
        height: int,
        nbframes: int = 1,
        frmRate: int = 1,
        intra_period: int = 1,
        bitdepth: int = 10,
        parallel_encoding: bool = False,
        hash_check: int = 0,
    ) -> List[Any]:
        level = 5.1 if nbframes > 1 else 6.2  # according to MPEG's anchor

        decodingRefreshType = 1 if intra_period >= 1 else 0

        base_cmd = [
            self.encoder_path,
            "-i",
            inp_yuv_path,
            "-c",
            self.cfg_file,
            "-q",
            qp,
            "-o",
            "/dev/null",
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            frmRate,
            "-ts",  # temporal subsampling
            1,
            "-dph",  # md5 has,
            hash_check,
            f"--Level={level}",
            f"--IntraPeriod={intra_period}",
            f"--DecodingRefreshType={decodingRefreshType}",
            "--InputChromaFormat=400",
            f"--InputBitDepth={bitdepth}",
            "--ConformanceWindowMode=1",  # needed?
        ]

        if parallel_encoding is False or nbframes <= (intra_period + 1):
            base_cmd.append(f"--BitstreamFile={bitstream_path}")
            base_cmd.append(f"--FramesToBeEncoded={nbframes}")
            cmd = list(map(str, base_cmd))
            self.logger.debug(cmd)
            return [cmd]

        num_parallels = round((nbframes / intra_period) + 0.5)

        list_of_num_of_frameSkip = []
        list_of_num_of_framesToBeEncoded = []
        total_num_frames_to_code = nbframes

        frameSkip = 0
        mnbframesToBeEncoded = intra_period + 1
        for _ in range(num_parallels):
            list_of_num_of_frameSkip.append(frameSkip)

            nbframesToBeEncoded = min(total_num_frames_to_code, mnbframesToBeEncoded)
            list_of_num_of_framesToBeEncoded.append(nbframesToBeEncoded)

            frameSkip += intra_period
            total_num_frames_to_code -= intra_period

        bitstream_path_p = Path(bitstream_path).parent
        file_stem = Path(bitstream_path).stem
        ext = Path(bitstream_path).suffix

        parallel_cmds = []
        for e, items in enumerate(
            zip(list_of_num_of_frameSkip, list_of_num_of_framesToBeEncoded)
        ):
            frameSkip, framesToBeEncoded = items
            sbitstream_path = (
                str(bitstream_path_p)
                + "/"
                + str(file_stem)
                + f"-part-{e:03d}"
                + str(ext)
            )

            pcmd = deepcopy(base_cmd)
            pcmd.append(f"--BitstreamFile={sbitstream_path}")
            pcmd.append(f"--FrameSkip={frameSkip}")
            pcmd.append(f"--FramesToBeEncoded={framesToBeEncoded}")

            cmd = list(map(str, pcmd))
            self.logger.debug(cmd)

            parallel_cmds.append(cmd)

        return parallel_cmds

    def get_merge_cmd(
        self,
        bitstream_path: str,
    ) -> List[Any]:
        pdir = Path(bitstream_path).parent
        fstem = Path(bitstream_path).stem
        ext = str(Path(bitstream_path).suffix)

        bitstream_lists = sorted(Path(pdir).glob(f"{fstem}-part-*{ext}"))

        cmd = [self.merger_path]
        for bpath in bitstream_lists:
            cmd.append(str(bpath))
        cmd.append(bitstream_path)

        cmd = list(map(str, cmd))
        self.logger.debug(cmd)
        return cmd, bitstream_lists

    def get_decode_cmd(self, yuv_dec_path: Path, bitstream_path: Path) -> List[Any]:
        cmd = [self.decoder_path, "-b", bitstream_path, "-o", yuv_dec_path]

        cl = list(map(str, cmd))
        self.logger.debug(cl)

        return [cl]

    def encode(
        self,
        x: Dict,
        file_prefix: str,
        last_input: bool,
    ) -> bool:
        # check items to register
        assert set(["frm", "chSize", "minv", "maxv"]).issubset(x)

        frmH, frmW = self.register_frame_to_input_buffer(x["frm"])
        self.register_minmax_to_buffer(x["minv"], x["maxv"])

        if last_input is False:
            return None

        nbframes = self.get_input_buffer_size()

        assert self.get_minmax_buffer_size() == nbframes

        frmRate = self.frame_rate if nbframes > 1 else 1
        intra_period = self.intra_period if nbframes > 1 else 1

        file_prefix = (
            f"{file_prefix}_{frmW}x{frmH}_{frmRate}fps_{self.bitdepth}bit_p400"
        )

        yuv_in_path = f"{file_prefix}_input.yuv"
        bitstream_path = f"{file_prefix}.bin"
        logpath = Path(f"{file_prefix}_{self.name}_enc.log")

        self.yuvio.setWriter(
            write_path=yuv_in_path,
            frmWidth=frmW,
            frmHeight=frmH,
        )

        for frame in self.input_buffer:
            self.yuvio.write_one_frame(frame)

        self.yuvio.closeWriter()

        cmds = self.get_encode_cmd(
            yuv_in_path,
            width=frmW,
            height=frmH,
            qp=self.qp,
            bitstream_path=bitstream_path,
            nbframes=nbframes,
            frmRate=frmRate,
            intra_period=intra_period,
            bitdepth=self.bitdepth,
            parallel_encoding=self.parallel_encoding,
            hash_check=self.hash_check,
        )

        start = time_measure()
        run_cmdline(cmds, logpath=logpath)
        enc_time = time_measure() - start
        self.logger.debug(f"enc_time:{enc_time}")

        if len(cmds) > 1:  # post parallel encoding
            cmd, list_of_bitstreams = self.get_merge_cmd(bitstream_path)
            inst_run_cmdline(cmd)

            if self.stash_outputs:
                for partial in list_of_bitstreams:
                    Path(partial).unlink()

        assert Path(
            bitstream_path
        ).is_file(), f"bitstream {bitstream_path} was not created"

        # adhoc method to make bitstream self-decodable
        _ = self.write_n_bit(self.bitdepth)
        _ = self.write_rft_chSize(x["chSize"])
        _ = self.write_packed_frame_size((frmH, frmW))
        _ = self.write_min_max_values()

        pre_info_bitstream = self.get_io_buffer_contents()
        inner_codec_bitstream = self.load_up_bitstream(bitstream_path)

        bitstream = pre_info_bitstream + inner_codec_bitstream

        if self.stash_outputs:
            Path(yuv_in_path).unlink()
            Path(bitstream_path).unlink()

        # [TODO]
        # to be compatible with the pipelines
        # per frame bits can be collected by parsing enc log to be more accurate
        avg_bytes_per_frame = len(bitstream) / nbframes
        all_bytes_per_frame = [avg_bytes_per_frame] * nbframes

        return {
            "bytes": all_bytes_per_frame,
            "bitstream": bitstream,
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

        yuv_dec_path = f"{file_prefix}_dec.yuv"
        bitstream_path = f"{file_prefix}_tmp.bin"
        logpath = Path(f"{file_prefix}_{self.name}_dec.log")

        with open(bitstream_path, "wb") as fw:
            fw.write(bitstream_fd.read())

        cmd = self.get_decode_cmd(
            bitstream_path=bitstream_path, yuv_dec_path=yuv_dec_path
        )

        start = time_measure()
        run_cmdline(cmd, logpath=logpath)
        dec_time = time_measure() - start
        self.logger.debug(f"dec_time:{dec_time}")

        self.yuvio.setReader(
            read_path=yuv_dec_path,
            frmWidth=frmW,
            frmHeight=frmH,
        )

        nbframes = get_filesize(yuv_dec_path) // (frmW * frmH * 2)
        assert nbframes == len(_min_max_buffer)

        rec_frames = []
        list_of_minv = []
        list_of_maxv = []
        for i in range(nbframes):
            rec_yuv = self.yuvio.read_one_frame(i)
            minv, maxv = _min_max_buffer[i]

            rec_frames.append(rec_yuv)
            list_of_minv.append(minv)
            list_of_maxv.append(maxv)

        if self.stash_outputs:
            Path(yuv_dec_path).unlink()
            Path(bitstream_path).unlink()

        return {
            "frm": rec_frames,
            "chSize": (chH, chW),
            "minv": list_of_minv,
            "maxv": list_of_maxv,
            "n_bit": n_bit,
        }
