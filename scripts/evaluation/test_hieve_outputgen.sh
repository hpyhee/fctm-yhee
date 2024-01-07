#!/bin/bash

export DEVICE=${DEVICE}
export DNNL_MAX_CPU_ISA=AVX2

DEVICE="cpu"
DATASET_DIR="/home/yhee/Nvme_4T_1/fcm_testdata"
OUTPUT_DIR="/home/yhee/Nvme_4T_1/proj/exps/fctm-etri"
VTM_PATH="/home/yhee/Nvme_4T_1/proj/VTM/VTM-12.0"
CV_PATH="/home/yhee/Nvme_4T_1/proj"
EXP_NAME="_PreBM"

Result_PATH=${OUTPUT_DIR}/split-inference-video/fctm${EXP_NAME}/MPEGHIEVE
echo $Result_PATH
ls $Result_PATH

python3 ${CV_PATH}/CompressAI-Vision/utils/fcm_cttc_output_gen.py -r ${OUTPUT_DIR}/split-inference-video/fctm${EXP_NAME}/MPEGHIEVE -dp ${DATASET_DIR}/HiEve_pngs -dn "HIEVE"
