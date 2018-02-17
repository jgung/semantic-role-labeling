#!/bin/bash

PROGRAM_NAME=$0

TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
DEVEL_SECTIONS=(24)

function usage {
    echo "usage: $PROGRAM_NAME [wsj_path] [output_path]"
    echo "  wsj_path       path to root directory of Penn Treebank dataset"
    echo "  output_path    output path to save results"
    exit 1
}

if [ "$#" -gt 1 ]; then
    if [ ! -d $1 ]; then
        echo "$1 does not exist."
        exit 1
    elif [ ! -d $1/parsed ]; then
        echo "Couldn't locate directory 'parsed' in $1. Make sure you have provided the correct directory."
        exit 1
    else
        WSJPATH=$1
    fi
    OUTPUT_PATH=$2
else
    usage
fi

CONLL05_PATH="$OUTPUT_PATH/conll05st-release"

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

retrieve_words() {
    section_type=$1
    declare -a sections=("${!2}")
    echo "Extracting words from PTB $section_type sections ${sections[@]}"
    checkdir ${CONLL05_PATH}/${section_type}/words
    for section in "${sections[@]}"
    do
        if [ ! -f "$CONLL05_PATH/${section_type}/words/${section_type}.$section.words.gz" ]; then
            echo ...${section}
            cat ${WSJPATH}/parsed/mrg/wsj/${section}/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | \
                gzip > "$CONLL05_PATH/${section_type}/words/${section_type}.$section.words.gz"
        else
            echo "Skipping retrieving words for section $section (file already exists)"
        fi
    done
}

export PERL5LIB="$OUTPUT_PATH/srlconll-1.1/lib:$PERL5LIB"
export PATH="$OUTPUT_PATH/srlconll-1.1/bin:$PATH"

checkdir ${OUTPUT_PATH}

checkdownload srlconll-1.1.tgz http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz
checkdownload conll05st-release.tar.gz http://www.lsi.upc.edu/~srlconll/conll05st-release.tar.gz
checkdownload conll05st-tests.tar.gz http://www.lsi.upc.edu/~srlconll/conll05st-tests.tar.gz

retrieve_words train TRAIN_SECTIONS[@] && retrieve_words devel DEVEL_SECTIONS[@]
