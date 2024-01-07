#!/bin/bash

# source ../venv/bin/activate

DEVICE="cpu"

export DEVICE=${DEVICE}
export DNNL_MAX_CPU_ISA=AVX2
#OIV6 HNU ENCODE TEST

DATASET_DIR="/home/ubuntu/data/vcm/FCTM/fcm_testdata"
OUTPUT_DIR="/home/ubuntu/data/vcm/FCTM/RSW/fctm"
VTM_PATH="/home/ubuntu/data/vcm/FCTM/RSW/VVCSoftware_VTM"

###########################################################################
# EXP CONDITION
EXP_NAME="_anchor"
###########################################################################
SEQs=(Traffic_2560x1600_30_val Kimono_1920x1080_24_val ParkScene_1920x1080_24_val Cactus_1920x1080_50_val BasketballDrive_1920x1080_50_val BasketballDrill_832x480_50_val BQTerrace_1920x1080_60_val BQSquare_416x240_60_val PartyScene_832x480_50_val RaceHorses_832x480_30_val RaceHorses_416x240_30_val BlowingBubbles_416x240_50_val BasketballPass_416x240_50_val BQMall_832x480_60_val)

for SEQ in "${SEQs[@]}";do
  if [ $SEQ == Traffic_2560x1600_30_val ];then
    QPs=(24 28 29 34)
  elif [ $SEQ == Kimono_1920x1080_24_val ];then
    QPs=(27 35 37 39)
  elif [ $SEQ == ParkScene_1920x1080_24_val ];then
    QPs=(18 22 32 36)
  elif [ $SEQ == Cactus_1920x1080_50_val ];then
    QPs=(41 47 49 51)
  elif [ $SEQ == BasketballDrive_1920x1080_50_val ];then
    QPs=(22 27 32 37)
  elif [ $SEQ == BasketballDrill_832x480_50_val ];then
    QPs=(22 27 32 39)
  elif [ $SEQ == BQTerrace_1920x1080_60_val ];then
    QPs=(22 27 29 32)
  elif [ $SEQ == BQSquare_416x240_60_val ];then
    QPs=(22 27 30 34)
  elif [ $SEQ == PartyScene_832x480_50_val ];then
    QPs=(20 30 32 39)
  elif [ $SEQ == RaceHorses_832x480_30_val ];then
    QPs=(26 32 37 41)
  elif [ $SEQ == RaceHorses_416x240_30_val ];then
    QPs=(22 27 37 41)
  elif [ $SEQ == BlowingBubbles_416x240_50_val ];then
    QPs=(27 35 36 37)
  elif [ $SEQ == BasketballPass_416x240_50_val ];then
    QPs=(27 32 35 39)
  elif [ $SEQ == BQMall_832x480_60_val ];then
    QPs=(27 32 37 39)
fi

for QP in "${QPs[@]}";do
bash sfu_hw_obj/fctm_eval_on_sfu_hw_obj.sh ${DATASET_DIR} ${VTM_PATH} ${OUTPUT_DIR} ${EXP_NAME} ${DEVICE} ${QP} ${SEQ} ""
done
done

