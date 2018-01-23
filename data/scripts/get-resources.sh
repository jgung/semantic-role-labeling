#!/bin/bash

VECTOR_PATH="data/vectors"
GLOVE_6B="$VECTOR_PATH/glove.6B.zip"

if [ ! -d ${VECTOR_PATH} ]; then
    mkdir -p ${VECTOR_PATH}
fi

wget -O ${GLOVE_6B} http://nlp.stanford.edu/data/glove.6B.zip
unzip ${GLOVE_6B} -d ${VECTOR_PATH}
rm ${GLOVE_6B}