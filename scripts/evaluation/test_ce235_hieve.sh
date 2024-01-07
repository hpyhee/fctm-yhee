#!/bin/bash

DEVICE="cpu"

export DEVICE=${DEVICE}
export DNNL_MAX_CPU_ISA=AVX2

DATASET_DIR="/home/yhee/Nvme_4T_1/fcm_testdata"
OUTPUT_DIR="/home/yhee/Nvme_4T_1/proj/exps/fctm-etri"
VTM_PATH="/home/yhee/Nvme_4T_1/proj/VTM/VTM-12.0"
CV_PATH="/home/yhee/Nvme_4T_1/proj"
EXP_NAME="_ce235_hieve"

bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 24 "13" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 25 "13" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 26 "13" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 27 "13" ""

bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 24 "16" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 25 "16" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 26 "16" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 27 "16" ""

bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 30 "2" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 31 "2" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 32 "2" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 33 "2" ""

bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 22 "17" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 23 "17" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 24 "17" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 25 "17" ""

bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 25 "18" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 26 "18" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 27 "18" ""
bash hieve/fctm_eval_on_hieve_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 28 "18" ""

python3 ${CV_PATH}/CompressAI-Vision/utils/fcm_cttc_output_gen.py -r ${OUTPUT_DIR}/split-inference-video/fctm${EXP_NAME}/MPEGHIEVE -dp ${DATASET_DIR}/HiEve_pngs -dn "HIEVE"
