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

DATASET_SRC="${FCM_TESTDATA}/HiEve_pngs"
CONF_NAME="eval_DI"
CMD="compressai-vision-eval"


declare -A intra_period_dict
declare -A fr_dict

intra_period_dict["2"]=32
fr_dict["2"]=30

intra_period_dict["17"]=32
fr_dict["17"]=30

intra_period_dict["18"]=32
fr_dict["18"]=30

intra_period_dict["13"]=32
fr_dict["13"]=30

intra_period_dict["16"]=32
fr_dict["16"]=30

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
	++vision_model.arch=jde_1088x608 \
        ++vision_model.jde_1088x608.splits="[105, 90, 75]" \
        ++dataset.type=TrackingDataset \
        ++dataset.datacatalog=MPEGHIEVE \
	++dataset.settings.patch_size="[608, 1088]" \
        ++dataset.config.root=${DATASET_SRC}/${SEQ} \
        ++dataset.config.imgs_folder=img1 \
       	++dataset.config.annotation_file=gt/gt.txt \
        ++dataset.config.dataset_name=mpeg-hieve-${SEQ} \
        ++evaluator.type=MOT-HIEVE-EVAL \
        ++codec.experiment=${EXPERIMENT} \
	++codec.enc_configs.cfg_file=${INNER_CODEC_PATH}'/cfg/encoder_lowdelay_P_vtm.cfg' \
        ++codec.enc_configs.frame_rate=${FRAME_RATE} \
        ++codec.enc_configs.intra_period=${INTRA_PERIOD} \
        ++codec.enc_configs.parallel_encoding=True \
        ++codec.enc_configs.qp=${QP} \
        ++codec.tools.inner_codec.stash_outputs=True \
	++codec.tools.feature_reduction.m65705.enabled=False \
        ++codec.tools.feature_reduction.m65181.enabled=False \
        ++codec.tools.feature_reduction.m65704-o.enabled=False \
        ++codec.tools.feature_reduction.m65704-t.enabled=True \
        ++codec.tools.feature_reduction.m65704-t.split_point=ALT1 \
	++codec.tools.inner_codec.enc_exe=${INNER_CODEC_PATH}'/bin/EncoderAppStatic'  \
        ++codec.tools.inner_codec.dec_exe=${INNER_CODEC_PATH}'/bin/DecoderAppStatic' \
        ++codec.tools.inner_codec.merge_exe=${INNER_CODEC_PATH}'/bin/parcatStatic' \
        ++codec.eval_encode='bitrate' \
        ++codec.verbosity=0 \
	++codec.device=${DEVICE} \
        ++misc.device=${DEVICE} \
