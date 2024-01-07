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
Functions to register new tools.
"""

from __future__ import annotations

__all__ = [
    "FT_REDUCTION",
    "FT_RESTORATION",
    "INNER_CODECS",
    "register_ft_reduction",
    "register_ft_restoration",
    "register_inner_codec",
]

from typing import Any, Callable, Dict, Type, TypeVar

import torch.nn as nn

FT_REDUCTION: Dict[str, Callable[..., nn.Module]] = {}
FT_RESTORATION: Dict[str, Callable[..., nn.Module]] = {}
INNER_CODECS: Dict[str, Callable[..., nn.Module]] = {}


TFTReduction_b = TypeVar("TFTReduction_b", bound=nn.Module)
TFTRestoration_b = TypeVar("TFTRestoration_b", bound=nn.Module)
TInnerCodec_b = TypeVar("TInnerCodec_b", bound=nn.Module)


def register_ft_reduction(name: str):
    """Decorator for registering a dataset."""

    def decorator(cls: Type[TFTReduction_b]) -> Type[TFTReduction_b]:
        FT_REDUCTION[name] = cls
        return cls

    return decorator


def register_ft_restoration(name: str):
    """Decorator for registering a dataset."""

    def decorator(cls: Type[FT_RESTORATION]) -> Type[FT_RESTORATION]:
        FT_RESTORATION[name] = cls
        return cls

    return decorator


def register_inner_codec(name: str):
    """Decorator for registering a vision model"""

    def decorator(cls: Type[TInnerCodec_b]) -> Type[TInnerCodec_b]:
        INNER_CODECS[name] = cls
        return cls

    return decorator
