#!/bin/bash

PROGRAM_NAME=$0
TRAIN_FILE="train-set.conll"
DEVEL_FILE="dev-set.conll"
MODE="word"
CORPUS="conll05"
EXT="conll"

function usage()
{
    echo "Train SRL model with CoNLL-formatted data. Can read from either CoNLL-2012 or CoNLL-2005 datasets."
    echo ""
    echo "$PROGRAM_NAME -i path/to/srl/data -o path/to/output/files"
    echo -e "\t-h --help"
    echo -e "\t-i --input\tPath to directory containing train/dev files"
    echo -e "\t-o --output\tPath to directory for output files used during training, such as vocabularies and checkpoints"
    echo -e "\t-c --config\t(Optional) .json file used to configure features and model hyper-parameters"
    echo -e "\t-m --mode\t(Optional) mode, '$MODE' by default, or 'phrase' for phrase-constrained model"
    echo -e "\t-t --train\t(Optional) training corpus file name, '$TRAIN_FILE' by default"
    echo -e "\t-v --valid\t(Optional) validation corpus file name, '$DEVEL_FILE' by default"
    echo -e "\t--corpus\t(Optional) Corpus type in [conll2012, conll05], '$CORPUS' by default"
}

while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -h|--help)
    usage
    exit
    ;;
    -i|--input)
    DATA_PATH=$2
    shift
    shift
    ;;
    -o|--output)
    OUTPUT_PATH=$2
    shift
    shift
    ;;
    -c|--config)
    CONFIG=$2
    shift
    shift
    ;;
    -m|--mode)
    MODE=$2
    shift
    shift
    ;;
    -t|--train)
    TRAIN_FILE=$2
    shift
    shift
    ;;
    -d|--valid|--dev)
    DEVEL_FILE=$2
    shift
    shift
    ;;
    --corpus)
    CORPUS=$2
    shift
    shift
    ;;
    *)
    echo "Unknown option: $1"
    usage
    exit 1
    ;;
esac
done

if [ -z "$DATA_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    usage
    exit
fi

if [ -z "$CONFIG" ]; then
    CONFIG="data/configs/he_acl_2017.json"
    echo "Using default config at $CONFIG since none was provided."
fi

extract_features() {
    OUTPUT_FILE="${OUTPUT_PATH%/}/${2%$EXT}pkl"
    if [ -f ${OUTPUT_FILE} ]; then
        echo "Skipping $OUTPUT_FILE since it already exists."
        return 0
    fi

    INPUT_FILE="${DATA_PATH%/}/$2"
    echo "Extracting features from data at $INPUT_FILE and saving to $OUTPUT_FILE"

    FEAT_ARGS="./srl/data/srl_feature_extractor.py \
        --mode $1 \
        --input $INPUT_FILE \
        --output $OUTPUT_FILE \
        --config $CONFIG \
        --vocab $VOCAB_PATH \
        --dataset $CORPUS \
        --ext $EXT"
    if [[ ${MODE} == word ]]; then
        python ${FEAT_ARGS}
    elif [[ ${MODE} == phrase ]]; then
        python ${FEAT_ARGS} --phrase_input "data/datasets/phrases/conll05/${2%$EXT}phrases" --phrase_ext phrases
    else
        echo "Unrecognized mode: $MODE"
        return 1
    fi

    if [[ $? != 0 ]]; then
        echo "The last command exited with a non-zero status" 1>&2;
        return 1
    fi
    return 0
}

train_model() {
    LOAD=""
    if [ -f "$OUTPUT_PATH/checkpoint" ]; then
        echo "Continuing training from checkpoint file at ${OUTPUT_PATH%/}/checkpoint"
        LOAD="--load $OUTPUT_PATH"
    fi
    python ./srl/srl_trainer.py \
        --save "$OUTPUT_PATH/model-checkpoint" \
        --train "$OUTPUT_PATH/${TRAIN_FILE%$EXT}pkl" \
        --valid "$OUTPUT_PATH/${DEVEL_FILE%$EXT}pkl" \
        --output "$OUTPUT_PATH/${DEVEL_FILE%$EXT}.predictions.txt" \
        --config ${CONFIG} \
        --vocab ${VOCAB_PATH} \
        --log "$OUTPUT_PATH/trainer-$MODE.log" \
        --script ./data/scripts/srl-eval.pl \
        ${LOAD}
}

VOCAB_PATH="$OUTPUT_PATH/vocab"

if [ ! -d ${VOCAB_PATH} ]; then
    mkdir -p ${VOCAB_PATH}
fi

export PYTHONPATH=${PYTHONPATH}:`pwd`

extract_features new ${TRAIN_FILE} && extract_features update ${DEVEL_FILE} && train_model