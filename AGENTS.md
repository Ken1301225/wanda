# Repository Guidelines

## Project Structure & Module Organization
Root entry points are `main.py` for LLaMA-family pruning and `main_opt.py` for OPT models. Shared pruning, data loading, and evaluation logic lives in `lib/` (`prune.py`, `eval.py`, `data.py`). Reproducible experiment scripts are in `scripts/`. Fine-tuning extensions are split into `lora_ft/` and `dense_ft/`. Vision-model pruning code is isolated in `image_classifiers/`. Keep large artifacts such as model weights, caches, and outputs outside git-tracked paths; the code defaults to `llm_weights/` and commonly writes results under `out/`.

## Build, Test, and Development Commands
Set up the reference environment from `INSTALL.md`:
```sh
conda create -n prune_llm python=3.9
conda activate prune_llm
```
Install PyTorch, Transformers, Datasets, and Accelerate exactly as documented before running experiments.

Common workflows:
```sh
python main.py --model decapoda-research/llama-7b-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/wanda/
bash scripts/llama_7b.sh
cd lora_ft && bash script.sh
cd image_classifiers && bash download_weights.sh
```
Use the shell scripts when reproducing paper results; use direct `python ...` commands for targeted debugging.

## Coding Style & Naming Conventions
This is a Python research codebase with simple script-style organization. Use 4-space indentation, `snake_case` for functions and variables, and lowercase module names. Follow existing CLI naming patterns such as `--sparsity_ratio` and `--prune_method`. Keep changes dependency-light and local to the relevant submodule instead of introducing broad abstractions. No formatter or linter is configured in-repo, so match surrounding style carefully.

## Testing Guidelines
There is no dedicated automated test suite. Validate changes with the smallest command that exercises the edited path, then record the exact command in your PR. For pruning changes, run a targeted `python main.py ...` or `python main_opt.py ...` smoke test. For LoRA or dense fine-tuning changes, use the commands in `lora_ft/script.sh` and verify perplexity/evaluation output. Include output paths and any metric changes in your notes.

## Commit & Pull Request Guidelines
Recent commits use short, lowercase, imperative subjects such as `add dense ft` and `add code for opt`. Keep commit messages concise and scoped. PRs should state what changed, why it changed, the exact reproduction or validation commands used, required hardware or model checkpoints, and before/after metrics when behavior changes. Link issues when applicable, and do not commit downloaded weights, cached datasets, or secrets.
