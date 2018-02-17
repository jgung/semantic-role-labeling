#!/bin/bash

PROGRAM_NAME=$0

DEVEL_SECTIONS=(24)
TEST_WSJ=(wsj)
TEST_BROWN=(brown)

function usage {
    echo "usage: $PROGRAM_NAME [input_path] [output_path] [sections]"
    echo "  input_path    path to output of download script"
    echo "  output_path   name of output training file"
    echo "  sections      optional path to newline-separated sections for training data"
    exit 1
}

if [ "$#" -gt 0 ]; then
    INPUT_PATH=$1
    OUTPUT_PATH=$2
    if [ "$#" -gt 2 ]; then
        echo "Reading training sections at $3."
        declare -a TRAIN_SECTIONS
        readarray -t TRAIN_SECTIONS < $3
    else
        TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
    fi
else
    usage
fi

CONLL05_PATH="$INPUT_PATH/conll05st-release"

checkdir() {
    if [ ! -d $1 ]; then
        mkdir -p $1
    fi
}

make_dataset() {
    # (adopted from make-trainset.sh in conll05st-release)

    # sections that are considered to generate data; section numbers should be sorted
    declare -a SECTIONS=("${!1}")
    # output file name
    OUTPUT_FILE=$2
    # train/devel/test.wsj/test.brown
    SECTION_PATH=$3
    # train/devel/test
    SECTION_TYPE=$4
    # syntax type (col2, col2h, upc, or cha)
    if [ "$#" -gt 4 ]; then
        SYNTAX=$5
    else
        SYNTAX=cha
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        ZCAT=gzcat
    else
        ZCAT=zcat
    fi

    OUTPUT_FILE=${INPUT_PATH}/${OUTPUT_FILE}.conll
    if [[ -f ${OUTPUT_FILE} ]]; then
        echo "Skipping processing, output file $OUTPUT_FILE already exists"
        return 0
    fi

    checkdir tmp
    for section in "${SECTIONS[@]}"
    do
        echo "Processing section ${section}"
        ${ZCAT} ${SECTION_PATH}/words/${SECTION_TYPE}.${section}.words.gz > tmp/$$.words
        ${ZCAT} ${SECTION_PATH}/props/${SECTION_TYPE}.${section}.props.gz > tmp/$$.props
        if [[ ${SECTION_TYPE} != "test" ]]; then
            ${ZCAT} ${SECTION_PATH}/synt.${SYNTAX}/${SECTION_TYPE}.${section}.synt.${SYNTAX}.gz > tmp/$$.synt
            ${ZCAT} ${SECTION_PATH}/senses/${SECTION_TYPE}.${section}.senses.gz > tmp/$$.senses
            ${ZCAT} ${SECTION_PATH}/ne/${SECTION_TYPE}.${section}.ne.gz > tmp/$$.ne
            paste -d ' ' tmp/$$.words tmp/$$.synt tmp/$$.ne tmp/$$.senses tmp/$$.props | gzip > tmp/$$.section.${section}.gz
        else
            paste -d ' ' tmp/$$.words tmp/$$.props | gzip > tmp/$$.section.${section}.gz
        fi
    done

    echo "Generating file $OUTPUT_FILE"
    ${ZCAT} tmp/$$.section* > ${OUTPUT_FILE}
    echo "Cleaning temporary files"
    rm -f tmp/$$*
}

make_dataset TRAIN_SECTIONS[@] ${OUTPUT_PATH} ${CONLL05_PATH}/train train
make_dataset DEVEL_SECTIONS[@] dev-set ${CONLL05_PATH}/devel devel
make_dataset TEST_WSJ[@] test-wsj ${CONLL05_PATH}/test.wsj test
make_dataset TEST_BROWN[@] test-brown ${CONLL05_PATH}/test.brown test
