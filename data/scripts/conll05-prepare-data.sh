#!/bin/bash

PROGRAM_NAME=$0

TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
DEVEL_SECTIONS=(24)
TEST_WSJ=(wsj)
TEST_BROWN=(brown)

function usage()
{
    echo "Prepare CoNLL-2005 data."
    echo ""
    echo "$PROGRAM_NAME -i path/to/conll05/download -o training-output-name"
    echo -e "\t-h --help"
    echo -e "\t-i --input\tPath to raw CoNLL 2005 data"
    echo -e "\t-o --output\t(Optional) Training data output name (default='train-set')"
    echo -e "\t-s --sections\t(Optional) path to training sections (newline-separated file)"
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
    INPUT_PATH=$2
    shift
    shift
    ;;
    -o|--output)
    OUTPUT_PATH=$2
    shift
    shift
    ;;
    -s|--sections)
    TRAIN_SECTIONS=($(awk -F= '{print $1}' $2))
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

if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    usage
    exit 1
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
