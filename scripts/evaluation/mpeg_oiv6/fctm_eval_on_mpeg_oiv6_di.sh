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

DATASET_SRC="${FCM_TESTDATA}/mpeg-oiv6"
CONF_NAME="eval_DI"
CMD="compressai-vision-eval"

declare -A network_model
declare -A task_type

network_model["mpeg-oiv6-detection"]="faster_rcnn_X_101_32x8d_FPN_3x"
task_type["mpeg-oiv6-detection"]="obj"

network_model["mpeg-oiv6-segmentation"]="mask_rcnn_X_101_32x8d_FPN_3x"
task_type["mpeg-oiv6-segmentation"]="seg"


NETWORK_MODEL=${network_model[${SEQ}]}
TASK_TYPE=${task_type[${SEQ}]}
INTRA_PERIOD=1
FRAME_RATE=1

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
        ++pipeline.type=image \
        ++paths._run_root=${OUTPUT_DIR} \
	++vision_model.arch=${NETWORK_MODEL} \
        ++dataset.type=Detectron2Dataset \
        ++dataset.datacatalog=MPEGOIV6 \
        ++dataset.config.root=${DATASET_SRC} \
        ++dataset.config.annotation_file=annotations/${SEQ}-coco.json \
        ++dataset.config.dataset_name=${SEQ} \
        ++evaluator.type=OIC-EVAL \
        ++codec.experiment=${EXPERIMENT} \
	++codec.enc_configs.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_intra_vtm.cfg' \
        ++codec.enc_configs.frame_rate=${INTRA_PERIOD} \
        ++codec.enc_configs.intra_period=${FRAME_RATE} \
        ++codec.enc_configs.parallel_encoding=False \
        ++codec.enc_configs.qp=${QP} \
        ++codec.tools.inner_codec.stash_outputs=True \
	++codec.tools.feature_reduction.m65705.enabled=False \
        ++codec.tools.feature_reduction.m65181.enabled=False \
        ++codec.tools.feature_reduction.m65704-o.enabled=True \
        ++codec.tools.feature_reduction.m65704-t.enabled=False \
	++codec.tools.feature_reduction.m65704-o.task=${TASK_TYPE} \
	++codec.tools.inner_codec.enc_exe=${INNER_CODEC_PATH}'/bin/EncoderAppStatic'  \
        ++codec.tools.inner_codec.dec_exe=${INNER_CODEC_PATH}'/bin/DecoderAppStatic' \
        ++codec.tools.inner_codec.merge_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
        ++codec.eval_encode='bpp' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device=${DEVICE} \
