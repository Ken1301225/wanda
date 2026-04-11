#!/bin/bash

# Set common variables
# Keep paths without trailing '/' to avoid dynamic-module cache key collisions.
model="/data1/ldk/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0"
sparsity_ratio=0.75
cuda_device=0,1
seed=0

# Set CUDA device visibility
export CUDA_HOME=/data1/ldk/env/dkllm_dsv2/
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=$cuda_device
export HF_DATASETS_CACHE="/data1/ldk/huggingface/datasets"
export HF_HUB_CACHE="/data1/ldk/huggingface/hub"


# Define function to run python command
run_python_command () {
    python main_dsv2.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --seed $seed \
    --save $3 \
    --save_model $4 \
    --nsamples 128 
}



echo "Running with wanda pruning method"
run_python_command "ablate_wanda_seq" "unstructured" "/data1/ldk/SPNN/deepseekv2/wanda/output8" "/data1/ldk/SPNN/deepseekv2/wanda/ckpt8"
# run_python_command "sparsegpt" "unstructured" "/data1/ldk/SPNN/deepseekv2/wanda/sparsegpt/output" "/data1/ldk/SPNN/deepseekv2/wanda/sparsegpt/ckpt"
# run_python_command "wanda" "unstructured" "/data1/ldk/SPNN/deepseekv2/wanda/output6" "/data1/ldk/SPNN/deepseekv2/wanda/ckpt6"
# run_python_command "wanda" "unstructured" 
echo "Finished wanda pruning method"

