#!/bin/bash

PROGRAM_NAME=$0
OUTPUT_PATH="data/datasets/conll2012"

function usage()
{
    echo "Download and prepare CoNLL-2012 data. Requires OntoNotes 5 (https://catalog.ldc.upenn.edu/ldc2013t19)."
    echo "Uses scripts and data from http://cemantix.org/data/ontonotes.html"
    echo ""
    echo "$PROGRAM_NAME -i path/to/ontonotes -o path/to/output/files"
    echo -e "\t-h --help"
    echo -e "\t-i --ontonotes\tPath to root directory of OntoNotes 5 release (LDC201319)"
    echo -e "\t-o --output\tOutput path to save downloads and results"
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
    ONTONOTES_PATH=$2
    shift
    shift
    ;;
    -o|--output)
    OUTPUT_PATH=$2
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

if [ -z "$ONTONOTES_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    usage
    exit 1
fi

checkdir() {
    if [ ! -d $1 ]; then
        mkdir -p $1
    fi
}

# Download data
checkdownload() {
    if [ ! -f "$OUTPUT_PATH/$1" ]; then
        wget -O "$OUTPUT_PATH/$1" $2
    else
        echo "File $1 already exists, skipping download"
    fi
    tar xf "$OUTPUT_PATH/$1" -C "$OUTPUT_PATH"
}

checkdir ${OUTPUT_PATH}

checkdownload conll-formatted-ontonotes-5.0-scripts.tar.gz http://ontonotes.cemantix.org/download/conll-formatted-ontonotes-5.0-scripts.tar.gz
checkdownload v12.tar.gz https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
mv ${OUTPUT_PATH}/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data ${OUTPUT_PATH}/conll-formatted-ontonotes-5.0/
${OUTPUT_PATH}/conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ${ONTONOTES_PATH}/data/files/data ${OUTPUT_PATH}/conll-formatted-ontonotes-5.0/

find ${OUTPUT_PATH}/conll-formatted-ontonotes-5.0/data/development -name '*conll' | xargs cat > ${OUTPUT_PATH}/dev-set.conll
find ${OUTPUT_PATH}/conll-formatted-ontonotes-5.0/data/train -name '*conll' | xargs cat > ${OUTPUT_PATH}/train-set.conll
find ${OUTPUT_PATH}/conll-formatted-ontonotes-5.0/data/test -name '*conll' | xargs cat > ${OUTPUT_PATH}/test-set.conll
find ${OUTPUT_PATH}/conll-formatted-ontonotes-5.0/data/conll-2012-test -name '*conll' | xargs cat > ${OUTPUT_PATH}/conll-2012-test.conll