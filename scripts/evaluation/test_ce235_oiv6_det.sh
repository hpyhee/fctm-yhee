#!/bin/bash

export DEVICE=${DEVICE}
export DNNL_MAX_CPU_ISA=AVX2

DEVICE="cpu"
DATASET_DIR="/home/yhee/Nvme_4T_1/fcm_testdata"
OUTPUT_DIR="/home/yhee/Nvme_4T_1/proj/exps/fctm-etri"
VTM_PATH="/home/yhee/Nvme_4T_1/proj/VTM/VTM-12.0"
CV_PATH="/home/yhee/Nvme_4T_1/proj"
EXP_NAME="_ce235_oiv_det"

###########################################################################
# EXP CONDITION
MAX_PARALLEL=2
FRAMES_TO_BE_CODED=50
TASK="detection"
###########################################################################

QPs=(29 31 32 37)
echo "" > run_${EXP_NAME}.txt
for QP in "${QPs[@]}";do
for SKIP_N_FRAME in {0..4950..50};do
echo mpeg_oiv6/fctm_eval_on_mpeg_oiv6_di.sh ${DATASET_DIR} \"${VTM_PATH}\" ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} ${QP} \"mpeg-oiv6-${TASK}\" \"++pipeline.codec.encode_only=True ++pipeline.codec.skip_n_frames=${SKIP_N_FRAME} ++pipeline.codec.n_frames_to_be_encoded=${FRAMES_TO_BE_CODED}\" >> run_${EXP_NAME}.txt
done
done

###########################################################################

#OIV6 ENCODE TEST
while read sh; do
echo $sh
sem --id $$ -j $MAX_PARALLEL bash $sh
done <run_${EXP_NAME}.txt
sem --id $$ --wait

#OIV6 DECODE TEST
for QP in "${QPs[@]}";do
bash mpeg_oiv6/fctm_eval_on_mpeg_oiv6_di.sh ${DATASET_DIR} "${VTM_PATH}" ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} ${QP} "mpeg-oiv6-${TASK}" "++pipeline.codec.decode_only=True"
done
wait

python3 ${CV_PATH}/CompressAI-Vision/utils/fcm_cttc_output_gen.py -r ${OUTPUT_DIR}/split-inference-image/fctm${EXP_NAME}/MPEGOIV6 -dp ${DATASET_DIR}/mpeg-oiv6 -dn "OIV6"


