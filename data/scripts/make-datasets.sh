#!/bin/bash

TEST_DIR=$1
PTB_DIR=$2
SCRIPTS_DIR=./data/scripts

/bin/bash ${SCRIPTS_DIR}/conll05-download-data.sh ${PTB_DIR} ${TEST_DIR}

/bin/bash ${SCRIPTS_DIR}/conll05-data.sh ${TEST_DIR} train-0 ${SCRIPTS_DIR}/splits/0.txt
/bin/bash ${SCRIPTS_DIR}/conll05-data.sh ${TEST_DIR} train-1 ${SCRIPTS_DIR}/splits/1.txt
/bin/bash ${SCRIPTS_DIR}/conll05-data.sh ${TEST_DIR} train-2 ${SCRIPTS_DIR}/splits/2.txt
/bin/bash ${SCRIPTS_DIR}/conll05-data.sh ${TEST_DIR} train-3 ${SCRIPTS_DIR}/splits/3.txt
/bin/bash ${SCRIPTS_DIR}/conll05-data.sh ${TEST_DIR} train-4 ${SCRIPTS_DIR}/splits/4.txt
/bin/bash ${SCRIPTS_DIR}/conll05-data.sh ${TEST_DIR} train-all ${SCRIPTS_DIR}/splits/all.txt