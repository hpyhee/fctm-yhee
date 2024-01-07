# * The copyright in this software is being made available under the BSD
# * License, included below. This software may be subject to other third party
# * and contributor rights, including patent rights, and no such rights are
# * granted under this license.
# *
# * Copyright (c) 2023, ISO/IEC
# * All rights reserved.
# *
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *
# *  * Redistributions of source code must retain the above copyright notice,
# *    this list of conditions and the following disclaimer.
# *  * Redistributions in binary form must reproduce the above copyright notice,
# *    this list of conditions and the following disclaimer in the documentation
# *    and/or other materials provided with the distribution.
# *  * Neither the name of the ISO/IEC nor the names of its contributors may
# *    be used to endorse or promote products derived from this software without
# *    specific prior written permission.
# *
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# * THE POSSIBILITY OF SUCH DAMAGE.


import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

package_name = "fctm"
version = "1.0.0.dev0"
git_hash = "unknown"

cwd = Path(__file__).resolve().parent

try:
    git_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
    )
except (FileNotFoundError, subprocess.CalledProcessError):
    pass


def write_version_file():
    path = cwd / package_name / "version.py"
    with path.open("w") as f:
        f.write(f'__version__ = "{version}"\n')
        f.write(f'git_version = "{git_hash}"\n')


write_version_file()

TEST_REQUIRES = ["pytest", "pytest-cov"]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    # "flake8",
    # "flake8-bugbear",
    # "flake8-comprehensions",
    "isort",
]


def get_extra_requirements():
    extras_require = {
        "test": TEST_REQUIRES,
        "dev": DEV_REQUIRES,
        "doc": ["sphinx", "sphinx-book-theme", "sphinxcontrib-mermaid", "Jinja2<3.1"],
    }
    extras_require["all"] = {req for reqs in extras_require.values() for req in reqs}
    return extras_require


setup(
    name="fctm",
    version=version,
    install_requires=["hydra-core", "omegaconf", "yuvio"],
    packages=find_packages(),
    # include_package_data=True,
    entry_points={
        "console_scripts": [
            "fctm_encode = fctm.apps.fctm_encoder:main",
            "fctm_decode = fctm.apps.fctm_decoder:main"
        ]
    },
    # metadata for upload to PyPI
    author="MPEG-WG4 Feature Coding for Machines (FCM)",
    author_email=["mpeg-wg4-fcm@lists.aau.at",
    ],
    description="Feature Compression Test Model (FCTM) is a reference software to develop FCM standard",
    extras_require=get_extra_requirements(),
    license="BSD 3-Clause Clear License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
