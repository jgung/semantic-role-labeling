#!/bin/bash

PROGRAM_NAME=$0

TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
DEVEL_SECTIONS=(24)
TEST_WSJ=(wsj)
TEST_BROWN=(brown)

SRLPATH="./data/datasets/conll05"
CONLL05_PATH="$SRLPATH/conll05st-release"

function usage {
    echo "usage: $PROGRAM_NAME [wsj_path]"
    echo "  wsj_path    path to root directory of Penn Treebank dataset"
    exit 1
}

if [ "$#" -gt 0 ]; then
    WSJPATH=$1
else
    usage
fi

checkdir() {
    if [ ! -d $1 ]; then
        mkdir -p $1
    fi
}

# Download data
checkdownload() {
    if [ ! -f "$SRLPATH/$1" ]; then
        wget -O "$SRLPATH/$1" $2
    else
        echo "File $1 already exists, skipping download"
    fi
    tar xf "$SRLPATH/$1" -C "$SRLPATH"
}

retrieve_words() {
    section_type=$1
    declare -a sections=("${!2}")
    echo "Retrieving words from PTB $section_type sections ${sections[@]}"
    checkdir ${CONLL05_PATH}/${section_type}/words
    for section in "${sections[@]}"
    do
        if [ ! -f "$CONLL05_PATH/${section_type}/words/${section_type}.$section.words.gz" ]; then
            echo ${section}
            cat ${WSJPATH}/parsed/mrg/wsj/${section}/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | \
                gzip > "$CONLL05_PATH/${section_type}/words/${section_type}.$section.words.gz"
        else
            echo "Skipping retrieving words for section $section (file already exists)"
        fi
    done
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

    OUTPUT_FILE=${SRLPATH}/${OUTPUT_FILE}.conll
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
    ${ZCAT} tmp/$$.section* > ${SRLPATH}/${OUTPUT_FILE}
    echo "Cleaning temporary files"
    rm -f tmp/$$*
}

export PERL5LIB="$SRLPATH/srlconll-1.1/lib:$PERL5LIB"
export PATH="$SRLPATH/srlconll-1.1/bin:$PATH"

checkdir ${SRLPATH}

checkdownload srlconll-1.1.tgz http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz
checkdownload conll05st-release.tar.gz http://www.lsi.upc.edu/~srlconll/conll05st-release.tar.gz
checkdownload conll05st-tests.tar.gz http://www.lsi.upc.edu/~srlconll/conll05st-tests.tar.gz

retrieve_words train TRAIN_SECTIONS[@] && retrieve_words devel DEVEL_SECTIONS[@]

make_dataset TRAIN_SECTIONS[@] train-set ${CONLL05_PATH}/train train
make_dataset DEVEL_SECTIONS[@] dev-set ${CONLL05_PATH}/devel devel
make_dataset TEST_WSJ[@] test-wsj ${CONLL05_PATH}/test.wsj test
make_dataset TEST_BROWN[@] test-brown ${CONLL05_PATH}/test.brown test
