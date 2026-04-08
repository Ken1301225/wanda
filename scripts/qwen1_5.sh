#!/bin/bash

# Set common variables
model="/data1/ldk/model/Qwen1.5/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9/"
sparsity_ratio=0.75
cuda_device=0,1
seed=42

# Set CUDA device visibility
export CUDA_HOME=/data1/ldk/env/dkllm
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=$cuda_device
export HF_DATASETS_CACHE="/data1/ldk/huggingface/datasets"
export HF_HUB_CACHE="/data1/ldk/huggingface/hub"


# Define function to run python command
run_python_command () {
    python main_qwen.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    # --save $3 \
    # --save_model $4 \
}



echo "Running with wanda pruning method"
# run_python_command "wanda" "unstructured" "/data1/ldk/SPNN/qwen1_5/wanda/output4/" "/data1/ldk/SPNN/qwen1_5/wanda/ckpt4/"
run_python_command "wanda" "unstructured" 
echo "Finished wanda pruning method"
