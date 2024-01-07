#!/bin/bash

DEVICE="cpu"

export DEVICE=${DEVICE}
export DNNL_MAX_CPU_ISA=AVX2
#OIV6 HNU ENCODE TEST

DATASET_DIR="/home/ubuntu/data/vcm/FCTM/fcm_testdata"
OUTPUT_DIR="/home/ubuntu/data/vcm/FCTM/RSW/fctm"
VTM_PATH="/home/ubuntu/data/vcm/FCTM/RSW/VVCSoftware_VTM"

###########################################################################
# EXP CONDITION
MAX_PARALLEL=32
EXP_NAME="_oiv_anchor_det_cttc"
FRAMES_TO_BE_CODED=50
TASK="segmentation"
###########################################################################
QPs=(27 32 37 39)
echo "" > run_${EXP_NAME}.txt
for QP in "${QPs[@]}";do
for SKIP_N_FRAME in {0..4950..50};do
echo mpeg_oiv6/fctm_eval_on_mpeg_oiv6.sh ${DATASET_DIR} \"${VTM_PATH}\" ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} ${QP} \"mpeg-oiv6-${TASK}\" \"++pipeline.codec.encode_only=True ++pipeline.codec.skip_n_frames=${SKIP_N_FRAME} ++pipeline.codec.n_frames_to_be_encoded=${FRAMES_TO_BE_CODED}\" >> run_${EXP_NAME}.txt
done
done

###########################################################################

#OIV6 HNU ENCODE TEST
while read sh; do
echo $sh
sem --id $$ -j $MAX_PARALLEL bash $sh
done <run_${EXP_NAME}.txt
sem --id $$ --wait

#OIV6 HNU DECODE TEST
for QP in "${QPs[@]}";do
bash mpeg_oiv6/fctm_eval_on_mpeg_oiv6.sh ${DATASET_DIR} "${VTM_PATH}" ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} ${QP} "mpeg-oiv6-${TASK}" "++pipeline.codec.decode_only=True"
done
wait


