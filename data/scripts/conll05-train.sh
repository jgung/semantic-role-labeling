#!/bin/bash

PROGRAM_NAME=$0
DATA_PATH=$1
OUTPUT_PATH=$2
TRAIN_FILE="train-set.conll"
DEVEL_FILE="dev-set.conll"

function usage {
    echo "usage: $PROGRAM_NAME [[[input] output] config]"
    echo "  input   root directory for SRL train/dev/test files"
    echo "  output  directory for output files used during training, such as the resulting models and checkpoints"
    echo "  config  (optional) json file used to configure features and network parameters"
    exit 1
}

if [ "$#" -gt 2 ]; then
    CONFIG=$3
elif [ "$#" -lt 2 ]; then
    usage
else
    CONFIG="data/configs/he_acl_2017.json"
    printf "Using default config at %s since none was provided.\n" ${CONFIG}
fi

extract_features() {
    if [ -f "$OUTPUT_PATH/$2.pkl" ]; then
        printf "Skipping %s since it already exists.\n" "$OUTPUT_PATH$2.pkl"
        return 0
    fi
    printf "Extracting features from data at %s and saving to %s\n" "$DATA_PATH$2" "$OUTPUT_PATH$2.pkl"
    python ./srl/data/srl_feature_extractor.py \
        --mode $1 \
        --input "$DATA_PATH/$2" \
        --output "$OUTPUT_PATH/$2.pkl" \
        --config ${CONFIG} \
        --vocab ${VOCAB_PATH} \
        --dataset conll05
    if [[ $? != 0 ]]; then
        echo "The last command exited with a non-zero status" 1>&2;
        return 1
    fi
    return 0
}

train_model() {
    python ./srl/srl_trainer.py \
        --save ${VOCAB_PATH} \
        --train "$OUTPUT_PATH/$TRAIN_FILE.pkl" \
        --valid "$OUTPUT_PATH/$DEVEL_FILE.pkl" \
        --config ${CONFIG} \
        --vocab ${VOCAB_PATH} \
        --script ./data/scripts/srl-eval.pl
}

VOCAB_PATH="$OUTPUT_PATH/vocab"

if [ ! -d ${VOCAB_PATH} ]; then
    mkdir -p ${VOCAB_PATH}
fi

export PYTHONPATH=${PYTHONPATH}:`pwd`

extract_features new ${TRAIN_FILE} && extract_features update ${DEVEL_FILE} && train_model