#!/bin/bash
set -euo pipefail

# Fast test helper for source probing and quick-save without prune/eval.
model="/data1/ldk/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0"
out="/data1/ldk/SPNN/deepseekv2/wanda/ckpt2_quick_test"

cd "$(dirname "$0")/.."

python main_dsv2_test.py \
  --mode probe \
  --model "$model" \
  --local-files-only

python main_dsv2_test.py \
  --mode quick-save \
  --model "$model" \
  --out "$out" \
  --save-tokenizer \
  --local-files-only

# Optional heavier test (loads and saves weights):
python main_dsv2_test.py \
  --mode quick-save \
  --model "$model" \
  --out "$out" \
  --save-tokenizer \
  --with-weights \
  --device-map auto \
  --torch-dtype bfloat16 \
  --local-files-only
