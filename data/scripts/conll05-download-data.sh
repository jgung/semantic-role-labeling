#!/bin/bash

PROGRAM_NAME=$0

TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
DEVEL_SECTIONS=(24)

function usage()
{
    echo "Download and prepare CoNLL-2005 data. Requires PTB dataset from https://catalog.ldc.upenn.edu/ldc99t42."
    echo ""
    echo "$PROGRAM_NAME -i path/to/ptb -o path/to/output/files"
    echo -e "\t-h --help"
    echo -e "\t-i --ptb\tPath to root directory of Penn TreeBank (LDC99T42)"
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
    PTB_PATH=$2
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

if [ -z "$PTB_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    usage
    exit 1
fi

if [ ! -d "$PTB_PATH" ]; then
    echo "Error: The provided path '$PTB_PATH' does not exist."
    exit 1
    usage
elif [ ! -d "$PTB_PATH/parsed" ]; then
    echo "Error: Couldn't locate directory 'parsed' in '$PTB_PATH'. Make sure you have provided the correct directory."
    usage
    exit 1
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
            cat ${PTB_PATH}/parsed/mrg/wsj/${section}/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | \
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
