#!/bin/bash

DEVICE="cpu"

export DEVICE=${DEVICE}
export DNNL_MAX_CPU_ISA=AVX2

DATASET_DIR="/home/yhee/Nvme_4T_1/fcm_testdata"
OUTPUT_DIR="/home/yhee/Nvme_4T_1/proj/exps/fctm-etri"
VTM_PATH="/home/yhee/Nvme_4T_1/proj/VTM/VTM-12.0"
CV_PATH="/home/yhee/Nvme_4T_1/proj"
EXP_NAME="_ce235_tvd"


bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 25 "TVD-01" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 27 "TVD-01" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 28 "TVD-01" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 31 "TVD-01" ""

bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 30 "TVD-02" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 31 "TVD-02" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 33 "TVD-02" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 34 "TVD-02" ""

bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 28 "TVD-03" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 30 "TVD-03" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 32 "TVD-03" ""
bash tvd/fctm_eval_on_tvd_tracking_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} 33 "TVD-03" ""

python3 ${CV_PATH}/CompressAI-Vision/utils/fcm_cttc_output_gen.py -r ${OUTPUT_DIR}/split-inference-video/fctm${EXP_NAME}/MPEGTVDTRACKING -dp ${DATASET_DIR}/tvd_tracking -dn "TVD"
