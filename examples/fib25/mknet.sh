#!/usr/bin/env bash

CONTAINER=funkey/mala:v0.1-pre1

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
    /bin/bash -c "export PYTHON_EGG_CACHE=/run; export PYTHONPATH=/custom:/src/malis:/src/waterz:/src/dvision:/src/augment:/src/caffe/python:/src
    /caffe/python/caffe
    :$PYTHONPATH; python -u mknet.py"