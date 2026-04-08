# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def _find_local_c4_file(filename):
    hub_cache = os.environ.get("HF_HUB_CACHE", "/data1/ldk/huggingface/hub")
    pattern = os.path.join(
        hub_cache,
        "datasets--allenai--c4",
        "snapshots",
        "*",
        "en",
        filename,
    )
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def _load_c4_split_from_local(split, filename):
    local_file = _find_local_c4_file(filename)
    if local_file is None:
        raise ValueError(
            f"Offline fallback failed: local C4 file not found for split={split}, filename={filename}."
        )
    return load_dataset("json", data_files={split: local_file}, split=split)
# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen,  tokenizer):
    train_file = 'en/c4-train.00000-of-01024.json.gz'
    val_file = 'en/c4-validation.00000-of-00008.json.gz'

    try:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': train_file}, split='train'
        )
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': val_file}, split='validation'
        )
    except ValueError as exc:
        if "Couldn't find cache for allenai/c4" not in str(exc):
            raise
        print("Falling back to local cached C4 snapshot files...")
        traindata = _load_c4_split_from_local('train', 'c4-train.00000-of-01024.json.gz')
        valdata = _load_c4_split_from_local('validation', 'c4-validation.00000-of-00008.json.gz')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  #构造语言模型的tar, 除了最后一个token其他都是-100, 只计算最后一个token的loss
        trainloader.append((inp, tar)) 

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
