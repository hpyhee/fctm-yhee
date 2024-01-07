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


from typing import List, Literal

import torch.nn as nn

from fctm.libs.regitsters import register_ft_reduction, register_ft_restoration

title = lambda a: str(a.__class__).split("<class '")[-1].split("'>")[0].split(".")[-1]

_OP_TYPE = Literal["enc", "dec"]


@register_ft_restoration("bypass")
@register_ft_reduction("bypass")
class base(nn.Module):
    """
    base input to output
    """

    def __init__(self, tag: str, device, **kwargs):
        super().__init__()
        self.tag = tag
        self.device = device

    @property
    def name(self):
        return f"{title(self)}({self.tag})"

    def to_device(self, x):
        return x.to(self.device)

    def forward(self, input):
        return input

    def inverse(self, input):
        return input

    def get_split_coding_info(self, tag: str, diag: str, op: _OP_TYPE):
        r"""split coding information details about weight address and interface
        Args:
            tag  (str): tool name
            diag (str): split point or task specific info to load corresponding information
            op   (str): operation point
        """
        split_db = {
            "m65705": {  # tool tag
                "obj": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65705_ifw_faster_rcnn_x101_139173657_s_a_fpn_enc-317e8fb8.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65705_ifw_faster_rcnn_x101_139173657_s_a_fpn_dec-adc67b68.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                },
                "seg": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65705_ifw_mask_rcnn_x101_139653917_s_a_fpn_enc-da979b00.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65705_ifw_mask_rcnn_x101_139653917_s_a_fpn_dec-62d38a75.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                },
            },
            "m65181": {  # tool tag
                "alt1": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65181_ifw_yolov3_jde_1088x608_s_a_alt1_enc-a248586c.pth.tar",
                        "split_if": {105: 128, 90: 256, 75: 512},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65181_ifw_yolov3_jde_1088x608_s_a_alt1_dec-d8dd691a.pth.tar",
                        "split_if": {105: 128, 90: 256, 75: 512},
                    },
                },
                "dn53": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65181_ifw_yolov3_jde_1088x608_s_a_dn53_enc-2533246e.pth.tar",
                        "split_if": {36: 256, 61: 512, 74: 1024},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65181_ifw_yolov3_jde_1088x608_s_a_dn53_dec-5999915e.pth.tar",
                        "split_if": {36: 256, 61: 512, 74: 1024},
                    },
                },
            },
            "m65704-o": {  # tool tag                                                                                 #Digital Insights
                "obj": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65704_ifw_faster_rcnn_x101_139173657_s_a_fpn_enc-317e8fb8.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65704_ifw_faster_rcnn_x101_139173657_s_a_fpn_dec-adc67b68.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                },
                "seg": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65704_ifw_mask_rcnn_x101_139653917_s_a_fpn_enc-da979b00.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65704_ifw_mask_rcnn_x101_139653917_s_a_fpn_dec-62d38a75.pth.tar",
                        "split_if": {"p2": 256, "p3": 256, "p4": 256, "p5": 256},
                    },
                },
            },
            "m65704-t": {  # tool tag                                                                                 #Digital Insights
                "alt1": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65704_ifw_yolov3_jde_1088x608_s_a_alt1_enc-a248586c.pth.tar",
                        "split_if": {105: 128, 90: 256, 75: 512},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65704_ifw_yolov3_jde_1088x608_s_a_alt1_dec-d8dd691a.pth.tar",
                        "split_if": {105: 128, 90: 256, 75: 512},
                    },
                },
                "dn53": {  # diag (task or split point)
                    "enc": {  # operation point
                        "weight": f"{tag}/m65704_ifw_yolov3_jde_1088x608_s_a_dn53_enc-2533246e.pth.tar",
                        "split_if": {36: 256, 61: 512, 74: 1024},
                    },
                    "dec": {  # operation point
                        "weight": f"{tag}/m65704_ifw_yolov3_jde_1088x608_s_a_dn53_dec-5999915e.pth.tar",
                        "split_if": {36: 256, 61: 512, 74: 1024},
                    },
                },
            },
        }

        assert tag.lower() in split_db.keys()
        tool_info = split_db[tag.lower()]

        assert diag.lower() in tool_info.keys()
        split_info = tool_info[diag.lower()]

        return split_info[op.lower()]

    @staticmethod
    def sort_by_shape(data: List, tShapes: List):
        itemized_data = {}
        for ft in data:
            assert ft.dim() == 4
            itemized_data[ft.shape[1:]] = ft.cpu()

        res = []
        for ts in tShapes:
            res.append(itemized_data[ts])

        return res
