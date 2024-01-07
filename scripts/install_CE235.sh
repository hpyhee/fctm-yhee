#!/usr/bin/env bash
#
# This clones and build model architectures and gets pretrained weights
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo
echo "Installing Feature Compression Test Model"
echo

pip3 install -e "${SCRIPT_DIR}/.."

echo 
echo "Overwriting black and isort libraries with specific version"
echo

pip3 install black==23.10.0
pip3 install isort==5.12.0

echo
echo "Copy fctm and CE235 configuration yaml to CompressAI-Vision"
echo

cp scripts/yaml/compressai_vision_fctm.yaml ../CompressAI-Vision/cfgs/codec/fctm.yaml
cp scripts/yaml/fctm_DI.yaml ../CompressAI-Vision/cfgs/codec/fctm_DI.yaml
cp scripts/yaml/eval_DI.yaml ../CompressAI-Vision/cfgs/eval_DI.yaml





