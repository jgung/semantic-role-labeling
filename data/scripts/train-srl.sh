#!/bin/bash

PROGRAM_NAME=$0
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
    echo -e "\t-t --train\t(Optional) training corpus file name"
    echo -e "\t-v --valid\t(Optional) validation corpus file name"
    echo -e "\t--test\t(Optional) test corpus file name in directory given by '-o' or '--output'"
    echo -e "\t--corpus\t(Optional) Corpus type in [conll2012, conll05], '$CORPUS' by default"
    echo -e "\t--custom\t(Optional) .json configuration for a custom corpus reader"
    echo -e "\t--mappings\t(Optional) predicate mappings file (3 column: 'lemma roleset mapped_value')"
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
    -d|-v|--valid|--dev)
    DEVEL_FILE=$2
    shift
    shift
    ;;
    --test)
    TEST_FILE=$2
    shift
    shift
    ;;
    --corpus)
    CORPUS=$2
    shift
    shift
    ;;
    --mappings)
    MAPPINGS=$2
    shift
    shift
    ;;
    --custom)
    CUSTOM_READER=$2
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

if [[ -z "$DATA_PATH" ]] || [[ -z "$OUTPUT_PATH" ]]; then
    usage
    exit
fi

if [[ -z "$CONFIG" ]]; then
    CONFIG="data/configs/he_acl_2017.json"
    echo "Using default config at $CONFIG since none was provided (use --config to specify one)"
fi

if [[ -z "$TEST_FILE" ]]; then
    if [[ -z "$TRAIN_FILE" ]]; then
        TRAIN_FILE="train-set.conll"
        echo "Using default train file name, $TRAIN_FILE, since none was provided (use --train to specify one)"
    fi
    if [[ -z "$DEVEL_FILE" ]]; then
        DEVEL_FILE="dev-set.conll"
        echo "Using default devel file name, $DEVEL_FILE, since none was provided (use --dev to specify one)"
    fi
fi

extract_features() {
    OUTPUT_FILE="${OUTPUT_PATH%/}/${2%$EXT}pkl"
    if [[ -f ${OUTPUT_FILE} ]]; then
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

    if [[ -n ${CUSTOM_READER} ]]; then
        echo "Using custom reader at '$CUSTOM_READER'"
        FEAT_ARGS="$FEAT_ARGS --custom $CUSTOM_READER"
    fi

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

MAPPING_ARG=""
if [[ -n "$MAPPINGS" ]]; then
    echo "Using predicate mappings file at $MAPPINGS"
    MAPPING_ARG="--mappings $MAPPINGS"
fi

train_model() {
    LOAD=""
    if [[ -f "$OUTPUT_PATH/checkpoint" ]]; then
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
        ${LOAD} \
        ${MAPPING_ARG}
}

test_model() {
    if [[ ! -f "$OUTPUT_PATH/checkpoint" ]]; then
        echo "Couldn't locate checkpoint file at $OUTPUT_PATH/checkpoint".
        return 1
    fi
    python ./srl/srl_trainer.py \
        --test "$OUTPUT_PATH/${TEST_FILE%$EXT}pkl" \
        --output "$OUTPUT_PATH/${TEST_FILE%$EXT}.predictions.txt" \
        --config ${CONFIG} \
        --vocab ${VOCAB_PATH} \
        --log "$OUTPUT_PATH/tester-$MODE.log" \
        --script ./data/scripts/srl-eval.pl \
        --load ${OUTPUT_PATH} \
        ${MAPPING_ARG}
}

VOCAB_PATH="$OUTPUT_PATH/vocab"

if [[ ! -d ${VOCAB_PATH} ]]; then
    mkdir -p ${VOCAB_PATH}
fi

export PYTHONPATH=${PYTHONPATH}:`pwd`

if [[ -n "$TRAIN_FILE" ]]; then
    extract_features new ${TRAIN_FILE} && extract_features update ${DEVEL_FILE} && train_model
fi

if [[ -n "$TEST_FILE" ]]; then
    extract_features load ${TEST_FILE} && test_model
fi
