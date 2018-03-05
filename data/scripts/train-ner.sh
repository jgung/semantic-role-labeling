#!/bin/bash

PROGRAM_NAME=$0
CORPUS="conll03"

function usage()
{
    echo "Train NER model with CoNLL-formatted data. Can read from either CoNLL-2003 or CoNLL-2012 datasets."
    echo ""
    echo "$PROGRAM_NAME -t path/to/ner/train.txt -v path/to/ner/valid.txt -o path/to/output/files"
    echo -e "\t-h --help"
    echo -e "\t-t --train\tTraining corpus file name"
    echo -e "\t-v --valid\tValidation corpus file name"
    echo -e "\t--test\t(Optional) test corpus file name"
    echo -e "\t-o --output\tPath to directory for output files used during training, such as vocabularies and checkpoints"
    echo -e "\t-c --config\t(Optional) .json file used to configure features and model hyper-parameters"
    echo -e "\t--corpus\t(Optional) Corpus type in [[conll03, conll2012]], '$CORPUS' by default"
}

while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -h|--help)
    usage
    exit
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
    *)
    echo "Unknown option: $1"
    usage
    exit 1
    ;;
esac
done

if [[ -z "$TEST_FILE" ]]; then
    if [[ -z "$TRAIN_FILE" ]] || [[ -z "$DEVEL_FILE" ]]; then
        usage
        exit 1
    fi
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    usage
    exit 1
fi

if [[ -z "$CONFIG" ]]; then
    CONFIG="data/configs/ner.json"
    echo "Using default config at $CONFIG since none was provided."
fi

extract_features() {
    FILE_NAME=$(basename $2)
    OUTPUT_FILE="${OUTPUT_PATH%/}/$FILE_NAME.pkl"
    if [[ -f ${OUTPUT_FILE} ]]; then
        echo "Skipping $OUTPUT_FILE since it already exists."
        return 0
    fi

    INPUT_FILE="$2"
    echo "Extracting features from data at $INPUT_FILE and saving to $OUTPUT_FILE"

    FEAT_ARGS="./srl/data/ner_feature_extractor.py \
        --mode $1 \
        --input $INPUT_FILE \
        --output $OUTPUT_FILE \
        --config $CONFIG \
        --vocab $VOCAB_PATH \
        --dataset $CORPUS"
    python ${FEAT_ARGS}

    if [[ $? != 0 ]]; then
        echo "The last command exited with a non-zero status" 1>&2;
        return 1
    fi
    return 0
}

train_model() {
    LOAD=""
    if [[ -f "$OUTPUT_PATH/checkpoint" ]]; then
        echo "Continuing training from checkpoint file at ${OUTPUT_PATH%/}/checkpoint"
        LOAD="--load $OUTPUT_PATH"
    fi
    TRAIN_FILE_NAME=$(basename "$TRAIN_FILE")
    DEV_FILE_NAME=$(basename "$DEVEL_FILE")
    python ./srl/ner_trainer.py \
        --save "$OUTPUT_PATH/model-checkpoint" \
        --train "$OUTPUT_PATH/${TRAIN_FILE_NAME}.pkl" \
        --valid "$OUTPUT_PATH/${DEV_FILE_NAME}.pkl" \
        --output "$OUTPUT_PATH/${DEV_FILE_NAME}.predictions.txt" \
        --config ${CONFIG} \
        --vocab ${VOCAB_PATH} \
        --log "$OUTPUT_PATH/ner-trainer.log" \
        --script ./data/scripts/conlleval.pl \
        ${LOAD}
}

test_model() {
    if [[ ! -f "$OUTPUT_PATH/checkpoint" ]]; then
        echo "Couldn't locate checkpoint file at $OUTPUT_PATH/checkpoint".
        return 1
    fi
    TEST_FILE_NAME=$(basename "$TEST_FILE")
    python ./srl/ner_trainer.py \
        --test "$OUTPUT_PATH/${TEST_FILE_NAME}.pkl" \
        --output "$OUTPUT_PATH/${TEST_FILE_NAME}.predictions.txt" \
        --config ${CONFIG} \
        --vocab ${VOCAB_PATH} \
        --log "$OUTPUT_PATH/ner-tester.log" \
        --script ./data/scripts/conlleval.pl \
        --load ${OUTPUT_PATH}
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
