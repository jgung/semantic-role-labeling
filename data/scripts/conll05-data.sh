#!/bin/bash

PROGRAM_NAME=$0

SRLPATH="./data/datasets/conll05"

function usage {
    echo "usage: $PROGRAM_NAME [ptb_path]"
    echo "  ptb_path    path to root directory of Penn Treebank dataset"
    exit 1
}

if [ "$#" -gt 0 ]; then
    if [ ! -d $1 ]; then
        echo "$1 does not exist."
        exit 1
    elif [ ! -d $1/parsed ]; then
        echo "Couldn't locate directory 'parsed' in $1. Make sure you have provided the correct directory."
        exit 1
    else
        PTB_DIR=$1
    fi
else
    usage
fi

SCRIPTS_DIR=./data/scripts

/bin/bash ${SCRIPTS_DIR}/conll05-download-data.sh ${PTB_DIR} ${SRLPATH}
/bin/bash ${SCRIPTS_DIR}/conll05-prepare-data.sh ${SRLPATH} train-set ${SCRIPTS_DIR}/splits/conll05-train-sections.txt