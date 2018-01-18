#!/usr/bin/env bash

CONTAINER=funkey/gunpowder:v0.3-pre5

NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

USER_ID=${UID}
USER_HOME=${HOME}

echo "Starting as user ${USER_ID} with home ${HOME}"

nvidia-docker pull ${CONTAINER}

NV_GPU=0 nvidia-docker run --rm \
    -u ${USER_ID} \
    -e HOME=${USER_HOME} \
    -v ${PWD}:/run \
    -v ${PWD}/../..:/custom \
    -w /run \
    --name ${NAME} \
    ${CONTAINER} \
    /bin/bash -c 'export PYTHONPATH="/custom:$PYTHONPATH"; python -u train.py 40000 0 tf'
