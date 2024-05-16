#!/bin/bash -x

pip install --upgrade pip
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
pip install "optimum[neuronx, diffusers]"
pip install diffusers==0.23.0
while true; do sleep 1000; done
