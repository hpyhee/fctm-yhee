#!/bin/bash

export DEVICE=${DEVICE}
export DNNL_MAX_CPU_ISA=AVX2

DEVICE="cpu"
DATASET_DIR="/home/yhee/Nvme_4T_1/fcm_testdata"
OUTPUT_DIR="/home/yhee/Nvme_4T_1/proj/exps/fctm-etri"
VTM_PATH="/home/yhee/Nvme_4T_1/proj/VTM/VTM-12.0"
CV_PATH="/home/yhee/Nvme_4T_1/proj"
EXP_NAME="_PreBM"


SEQs=(Traffic_2560x1600_30_val Kimono_1920x1080_24_val ParkScene_1920x1080_24_val Cactus_1920x1080_50_val BasketballDrive_1920x1080_50_val BasketballDrill_832x480_50_val BQTerrace_1920x1080_60_val BQSquare_416x240_60_val PartyScene_832x480_50_val RaceHorses_832x480_30_val RaceHorses_416x240_30_val BlowingBubbles_416x240_50_val BasketballPass_416x240_50_val BQMall_832x480_60_val)

for SEQ in "${SEQs[@]}";do
  if [ $SEQ == Traffic_2560x1600_30_val ];then
    QPs=(22 24 27 30)
  elif [ $SEQ == Kimono_1920x1080_24_val ];then
    QPs=(35 36 37 38)
  elif [ $SEQ == ParkScene_1920x1080_24_val ];then
    QPs=(24 27 30 32)
  elif [ $SEQ == Cactus_1920x1080_50_val ];then
    QPs=(41 44 45 47)
  elif [ $SEQ == BasketballDrive_1920x1080_50_val ];then
    QPs=(22 24 25 27)
  elif [ $SEQ == BasketballDrill_832x480_50_val ];then
    QPs=(24 27 30 32)
  elif [ $SEQ == BQTerrace_1920x1080_60_val ];then
    QPs=(20 24 25 30)
  elif [ $SEQ == BQSquare_416x240_60_val ];then
    QPs=(20 24 27 30)
  elif [ $SEQ == PartyScene_832x480_50_val ];then
    QPs=(24 27 30 32)
  elif [ $SEQ == RaceHorses_832x480_30_val ];then
    QPs=(24 27 30 32)
  elif [ $SEQ == RaceHorses_416x240_30_val ];then
    QPs=(24 27 28 32)
  elif [ $SEQ == BlowingBubbles_416x240_50_val ];then
    QPs=(24 27 35 37)
  elif [ $SEQ == BasketballPass_416x240_50_val ];then
    QPs=(27 32 35 37)
  elif [ $SEQ == BQMall_832x480_60_val ];then
    QPs=(27 32 37 39)
fi

for QP in "${QPs[@]}";do
bash sfu_hw_obj/fctm_eval_on_sfu_hw_obj_di.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} ${QP} ${SEQ} ""
done
done

python3 ${CAVison_DIR}/CompressAI-Vision/utils/fcm_cttc_output_gen.py -r ${OUTPUT_DIR}/split-inference-video/fctm${EXP_NAME}/SFUHW -dp ${DATASET_DIR}/SFU_HW_Obj -dn "SFU"

