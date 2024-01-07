#!/usr/bin/env bash
set -eu
export DNNL_MAX_CPU_ISA=AVX2

FCM_TESTDATA=$1
INNER_CODEC_PATH=$2
OUTPUT_DIR=$3
EXPERIMENT=$4
DEVICE=$5
QP=$6
SEQ=$7
PIPELINE_PARAMS=$8

DATASET_SRC="${FCM_TESTDATA}/SFU_HW_Obj"
CONF_NAME="eval_fctm"
CMD="compressai-vision-eval"


declare -A intra_period_dict
declare -A fr_dict

intra_period_dict["Traffic_2560x1600_30_val"]=32
fr_dict["Traffic_2560x1600_30_val"]=30

intra_period_dict["Kimono_1920x1080_24_val"]=32
fr_dict["Kimono_1920x1080_24_val"]=24

intra_period_dict["ParkScene_1920x1080_24_val"]=32
fr_dict["ParkScene_1920x1080_24_val"]=24

intra_period_dict["Cactus_1920x1080_50_val"]=64
fr_dict["Cactus_1920x1080_50_val"]=50

intra_period_dict["BasketballDrive_1920x1080_50_val"]=64
fr_dict["BasketballDrive_1920x1080_50_val"]=50

intra_period_dict["BasketballDrill_832x480_50_val"]=64
fr_dict["BasketballDrill_832x480_50_val"]=50

intra_period_dict["BQTerrace_1920x1080_60_val"]=64
fr_dict["BQTerrace_1920x1080_60_val"]=60

intra_period_dict["BQSquare_416x240_60_val"]=64
fr_dict["BQSquare_416x240_60_val"]=60

intra_period_dict["PartyScene_832x480_50_val"]=64
fr_dict["PartyScene_832x480_50_val"]=50

intra_period_dict["RaceHorses_832x480_30_val"]=32
fr_dict["RaceHorses_832x480_30_val"]=30

intra_period_dict["RaceHorses_416x240_30_val"]=32
fr_dict["RaceHorses_416x240_30_val"]=30

intra_period_dict["BlowingBubbles_416x240_50_val"]=64
fr_dict["BlowingBubbles_416x240_50_val"]=50

intra_period_dict["BasketballPass_416x240_50_val"]=64
fr_dict["BasketballPass_416x240_50_val"]=50

intra_period_dict["BQMall_832x480_60_val"]=64
fr_dict["BQMall_832x480_60_val"]=60

INTRA_PERIOD=${intra_period_dict[${SEQ}]}
FRAME_RATE=${fr_dict[${SEQ}]}

echo "============================== RUNNING FCTM + COMPRESSAI-VISION =================================="
echo "Datatset location:  " ${FCM_TESTDATA}
echo "Output directory:   " ${OUTPUT_DIR}
echo "Experiment folder:  " "fctm"${EXPERIMENT}
echo "Running Device:     " ${DEVICE}
echo "Input sequence:     " ${SEQ}
echo "Seq. Framerate:     " ${FRAME_RATE}
echo "QP for Inner Codec: " ${QP}
echo "Intra Period for Inner Codec: "${INTRA_PERIOD}
echo "Other Parameters:   " ${PIPELINE_PARAMS}
echo "=================================================================================================="

${CMD} --config-name=${CONF_NAME}.yaml ${PIPELINE_PARAMS} \
        ++pipeline.type=video \
        ++paths._run_root=${OUTPUT_DIR} \
	++vision_model.arch=faster_rcnn_X_101_32x8d_FPN_3x \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=SFUHW \
        ++dataset.config.root=${DATASET_SRC}/${SEQ} \
        ++dataset.config.annotation_file=annotations/${SEQ}.json \
        ++dataset.config.dataset_name=sfu-hw-${SEQ} \
        ++evaluator.type=COCO-EVAL \
	++evaluator.eval_criteria=AP50 \
        ++codec.experiment=${EXPERIMENT} \
	++codec.enc_configs.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_lowdelay_P_vtm.cfg' \
        ++codec.enc_configs.frame_rate=${FRAME_RATE} \
        ++codec.enc_configs.intra_period=${INTRA_PERIOD} \
        ++codec.enc_configs.parallel_encoding=True \
        ++codec.enc_configs.qp=${QP} \
        ++codec.tools.inner_codec.stash_outputs=True \
	++codec.tools.feature_reduction.m65705.enabled=True \
        ++codec.tools.feature_reduction.m65181.enabled=False \
	++codec.tools.feature_reduction.m65705.task='obj' \
	++codec.tools.inner_codec.enc_exe=${INNER_CODEC_PATH}'/bin/EncoderAppStatic'  \
        ++codec.tools.inner_codec.dec_exe=${INNER_CODEC_PATH}'/bin/DecoderAppStatic' \
        ++codec.tools.inner_codec.merge_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
        ++codec.eval_encode='bitrate' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device=${DEVICE} \
