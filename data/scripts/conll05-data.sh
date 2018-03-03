#!/bin/bash

PROGRAM_NAME=$0
OUTPUT_PATH="data/datasets/conll05"
SCRIPTS_PATH="data/scripts"
TRAIN="train-set"
SECTIONS="${SCRIPTS_PATH}/splits/conll05-train-sections.txt"

function usage()
{
    echo "Download and prepare CoNLL-2005 training/test data. Requires PTB dataset from https://catalog.ldc.upenn.edu/ldc99t42."
    echo ""
    echo "$PROGRAM_NAME -i path/to/ptb"
    echo -e "\t-h --help"
    echo -e "\t-i --ptb\tPath to root directory of Penn TreeBank (LDC99T42)"
    echo -e "\t-o --output\t(Optional) Output path, $OUTPUT_PATH by default"
    echo -e "\t-s --scripts\t(Optional) Path to necessary scripts, $SCRIPTS_PATH by default"
    echo -e "\t-t --train\t(Optional) Training file name, $TRAIN by default"
    echo -e "\t--sections\t(Optional) Sections file, $SECTIONS by default"
}

while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -h|--help)
    usage
    exit
    ;;
    -i|--ptb)
    PTB_PATH=$2
    shift
    shift
    ;;
    -o|--output)
    OUTPUT_PATH=$2
    shift
    shift
    ;;
    -s|--scripts)
    SCRIPTS_PATH=$2
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

if [ -z "$PTB_PATH" ]; then
    usage
    exit 1
fi

/bin/bash ${SCRIPTS_PATH}/conll05-download-data.sh -i ${PTB_PATH} -o ${OUTPUT_PATH}
/bin/bash ${SCRIPTS_PATH}/conll05-prepare-data.sh -i ${OUTPUT_PATH} -o ${TRAIN} -s ${SECTIONS}