#!/bin/bash -x
while true; do sleep 1000; done
pip install --upgrade pip
#pip install diffusers==0.20.2 transformers==4.33.1 accelerate==0.22.0 safetensors==0.3.1 matplotlib Pillow ipython -U
#python /sd2_512_compile.py
#tar -czvf /${COMPILER_WORKDIR_ROOT}/${MODEL_FILE}.tar.gz /${COMPILER_WORKDIR_ROOT}/
#aws s3 cp /${COMPILER_WORKDIR_ROOT}/${MODEL_FILE}.tar.gz s3://${BUCKET}/${MODEL_FILE}.tar.gz
pip install --user --upgrade /neuronx_cc-0.0.0.0.dev0+75780daff7-cp310-cp310-linux_x86_64.whl
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install git+https://github.com/aws-shchung/optimum-neuron.git@main --force-reinstall --no-deps
python compile-sd-optimum-neuron.py
