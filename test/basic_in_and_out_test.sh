#!/bin/bash
set -e

fctm_encode --config-name=example_encoder.yaml \
            ++inspection_mode=True \
            ++paths._run_root="./test_output" \
            ++codec.input_feature_tensors="./test/samples/input_sa_PLYR.h5" \
            ++codec.tools.inner_codec.type=bypass \
            ++codec.verbosity=1 \
            ++codec.tools.feature_reduction.m65705.enabled=True \
            ++codec.tools.feature_reduction.m65181.enabled=False \

fctm_decode --config-name=example_decoder.yaml \
            ++inspection_mode=True \
            ++paths._run_root="./test_output" \
            ++codec.tools.inner_codec.type=bypass \
            ++codec.verbosity=1 \
            ++codec.input_feature_tensors="./test_output/fctm/default-input_sa_PLYR.bin" \
            ++codec.tools.feature_reduction.m65705.enabled=True \
            ++codec.tools.feature_reduction.m65181.enabled=False \

fctm_encode --config-name=example_encoder.yaml \
            ++inspection_mode=True \
            ++paths._run_root="./test_output" \
            ++codec.input_feature_tensors="./fctm/test/samples/input_sa_DN53.h5" \
            ++codec.tools.inner_codec.type=bypass \
            ++codec.verbosity=1 \
            ++codec.tools.feature_reduction.m65705.enabled=False \
            ++codec.tools.feature_reduction.m65181.enabled=True \
            ++codec.tools.feature_reduction.m65181.split_point=DN53 \

fctm_decode --config-name=example_decoder.yaml \
            ++inspection_mode=True \
            ++paths._run_root="./test_output" \
            ++codec.tools.inner_codec.type=bypass \
            ++codec.input_feature_tensors="./test_output/fctm/default-input_sa_DN53.bin" \
            ++codec.verbosity=1 \
            ++codec.tools.feature_reduction.m65705.enabled=False \
            ++codec.tools.feature_reduction.m65181.enabled=True \
            ++codec.tools.feature_reduction.m65181.split_point=DN53 \

fctm_encode --config-name=example_encoder.yaml \
            ++inspection_mode=True \
            ++paths._run_root="./test_output" \
            ++codec.input_feature_tensors="./fctm/test/samples/input_sa_ALT1.h5" \
            ++codec.tools.inner_codec.type=bypass \
            ++codec.verbosity=1 \
            ++codec.tools.feature_reduction.m65705.enabled=False \
            ++codec.tools.feature_reduction.m65181.enabled=True \
            ++codec.tools.feature_reduction.m65181.split_point=ALT1 \
        

fctm_decode --config-name=example_decoder.yaml \
            ++inspection_mode=True \
            ++paths._run_root="./test_output" \
            ++codec.tools.inner_codec.type=bypass \
            ++codec.verbosity=1 \
            ++codec.input_feature_tensors="./test_output/fctm/default-input_sa_ALT1.bin \
            ++codec.tools.feature_reduction.m65705.enabled=False \
            ++codec.tools.feature_reduction.m65181.enabled=True \
            ++codec.tools.feature_reduction.m65181.split_point=ALT1 \