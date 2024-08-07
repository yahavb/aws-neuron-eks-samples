
* Fork https://github.com/aws-neuron/aws-neuron-eks-samples/ and populate the `GITHUB_USER`.
* Export the following variables
```bash
export CLUSTER_NAME=yahavb-neuron-demo
export CF_STACK=sd21-neuron-image-pipeline
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --output text --query Account)
export AWS_REGION=us-west-2
export BASE_IMAGE_AMD_XLA_TAG=2.1.2-neuronx-py310-sdk2.19.0-ubuntu20.04
export IMAGE_AMD_XLA_TAG=neuron2.19
export BASE_REPO=stablediffusion
export BASE_TAG=multiarch-ubuntu
export BASE_AMD_TAG=amd64
export GITHUB_BRANCH=master
export NEURON_DLC_IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.19.0-ubuntu20.04
export GITHUB_USER=yahavb
export GITHUB_REPO=aws-neuron-eks-samples
export MODEL_DIR=sd21_compile_dir
export MODEL_FILE=stable-diffusion-2-1-base
export BUCKET=sdinfer
```

```bash
cd ci-build
./deploy-pipeline.sh
```
