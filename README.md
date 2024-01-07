# Introduction

This project provides Core Experiment CE2.3.6 for Feature Compression Test Model (FCTM).

# Our Computing Environment

- AMD Ryzen Threadripper PRO 3975WX 32-Cores 32core, 64thread
- NVIDIA RTX A5000
- Ubuntu 20.04.6
- GCC 9.4.0
- CUDA 11.8
- Nvidia Driver 545.23.08
-   Python 3.8.10
-   Numpy 1.24.4
-   Torch 2.0.0+cu118
-   Pandas 2.0.3
-   PIL 9.5.0 
-   GNU 'parallel' utility (for parallel VTM, inferencing execution)
-   ffmpeg 4.2.2
-   VTM-12.0 software

# Installation

Before get started locally and install our CE software, please make sure FCTM is installed and feature anchor results can be performed correctly. More details of FCTM installation can be checked in [README.md](http://mpegx.int-evry.fr/software/MPEG/Video/fcm/fctm/-/blob/main/README.md).

Our CE software operates within a similar virtual environment to FCTM.

First create a [virtual environment](https://docs.python.org/3.8/library/venv.html) with python==3.8.13:

```
python3.8.13 -m venv venv
source ./venv/bin/activate
pip install -U pip
```

Clone our CE software for the CE test ([FCM] CE 2.3.6 , Doc. m65704) at the ce236 branch from the fctm-yhee.git, and CompressAI-Vision with a tag of v1.1.3. Note that FCTM installs CompressAI-Vision with a tag of v1.1.2 (or latest) as a dependent library. Here, we install v1.1.3 that resolves minor issues checked in v1.1.2.

```
git clone --depth 1 --branch v1.1.3 https://github.com/InterDigitalInc/CompressAI-Vision.git 
git clone -b ce236 --single-branch https://mpeg.expert/software/hpyhee/fctm-yhee.git
```

Then, CompressAI-Vision should be installed and our CE software can be installed by: 

```
bash scripts/install_ce236.sh
```

## Evaluation

FCM CTTC Dataset and Dataset validation are the same as FCTM implementation. More details of this part can be checked in [README.md](http://mpegx.int-evry.fr/software/MPEG/Video/fcm/fctm/-/blob/main/README.md).

Example scripts for evaluation of CE2.3.6 on CTTC dataset can be found at [here](scripts/evaluation/)

```
evaluation/
├── hieve
│   └── fctm_eval_on_hieve_tracking_di.sh
├── mpeg_oiv6
│   └── fctm_eval_on_mpeg_oiv6_di.sh
├── sfu_hw_obj
│   └── fctm_eval_on_sfu_hw_obj_di.sh
└── tvd
    └── fctm_eval_on_tvd_tracking_di.sh
```

We provide the scripts to perform all performance points for each dataset. 

For the usage of the evaluation scripts, 
```
bash scripts/test_hieve.sh
bash scripts/test_sfu.sh
bash scripts/test_tvd.sh
bash scripts/test_oiv6_det.sh
bash scripts/test_oiv6_seg.sh
```

Please edit the evaluation scripts with your own working directory:
```
DATASET_DIR="/local/path/to/fcm_testdata"
OUTPUT_DIR="/local/path/to/output"
VTM_PATH="/local/path/to/VTM-12.0"
CV_PATH="/local/path/to/CompressAI-Vision"
EXP_NAME="name_of_experiments"
```
For example:

- DATASET_DIR="/home/yhee/Nvme_4T_1/fcm_testdata"
- OUTPUT_DIR="/home/yhee/Nvme_4T_1/proj/exps/fctm-etri"
- VTM_PATH="/home/yhee/Nvme_4T_1/proj/VTM/VTM-12.0"
- CV_PATH="/home/yhee/Nvme_4T_1"
- EXP_NAME="_oiv6_det_ce236"

All output files and results are stored in the directory of OUTPUT_DIR

Note that the EXP_NAME for OIV6 detection should be different from EXP_NAME for OIV6 segmentation.

Please edit the number of thread to run in parallel for test_oiv6_det.sh and test_oiv6_seg.sh:
```
MAX_PARALLEL="number of thread"
```

# Related links
 * [FCTM reference software](http://mpegx.int-evry.fr/software/MPEG/Video/fcm/fctm)
