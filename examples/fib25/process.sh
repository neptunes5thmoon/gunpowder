#!/usr/bin/env bash

NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

USER_ID=${UID}
USER_HOME=${HOME}

echo "Starting as user ${USER_ID} with home ${HOME}"

NV_GPU=0 nvidia-docker run --rm \
    -u ${USER_ID} \
    -e HOME=${USER_HOME} \
    -v ${PWD}:/run \
    -v ${PWD}/../..:/custom \
    -w /run \
    --name ${NAME} \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c 'export PYTHONPATH="/custom:$PYTHONPATH"; python -u process.py 0 tf'
