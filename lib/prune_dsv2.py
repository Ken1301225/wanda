import torch
import torch.nn as nn

from .ablate import AblateGPT
from .data import get_loaders
from .layerwrapper import WrappedGPT
from .sparsegpt import SparseGPT


def find_layers(module, layers=[nn.Linear], name=""):
    if type(module) in layers and ".experts." in name:
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def _get_hf_device_map(model):
    return getattr(model, "hf_device_map", {})


def _move_to_device(value, device):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _get_layer_kwargs(attention_mask=None, position_ids=None, position_embeddings=None):
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    return kwargs


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            weight = subset[name].weight.data
            zeros = (weight == 0).sum().item()
            params = weight.numel()

            count += zeros
            total_params += params
            sub_count += zeros
            sub_params += params

        if sub_params == 0:
            print(f"layer {i} sparsity n/a (no expert linear layers)")
            continue

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    if total_params == 0:
        return 0.0
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    hf_device_map = _get_hf_device_map(model)
    if "model.embed_tokens" in hf_device_map:
        device = hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, position_embeddings


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

    for layer in layers:
        subset = find_layers(layer)

        for name in subset:
            weight = subset[name].weight.data
            weight_metric = torch.abs(weight)
            if prune_n != 0:
                weight_mask = torch.zeros_like(weight) == 1
                for ii in range(weight_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = weight_metric[:, ii : (ii + prune_m)].float()
                        weight_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(weight_metric.flatten().cuda())[0][int(weight.numel() * args.sparsity_ratio)].cpu()
                weight_mask = weight_metric <= thresh

            weight[weight_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    layers = model.model.layers
    hf_device_map = _get_hf_device_map(model)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in hf_device_map:
            dev = hf_device_map[f"model.layers.{i}"]
            inps = _move_to_device(inps, dev)
            outs = _move_to_device(outs, dev)
            attention_mask = _move_to_device(attention_mask, dev)
            position_ids = _move_to_device(position_ids, dev)
            position_embeddings = _move_to_device(position_embeddings, dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        layer_kwargs = _get_layer_kwargs(attention_mask, position_ids, position_embeddings)
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        for handle in handles:
            handle.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            weight_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            weight_mask = torch.zeros_like(weight_metric) == 1
            if prune_n != 0:
                for ii in range(weight_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = weight_metric[:, ii : (ii + prune_m)].float()
                        weight_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(weight_metric, dim=-1, stable=True)

                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = weight_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    weight_mask, cur_sparsity = return_given_alpha(alpha, sort_res, weight_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        weight_mask, cur_sparsity = return_given_alpha(alpha, sort_res, weight_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, : int(weight_metric.shape[1] * args.sparsity_ratio)]
                    weight_mask.scatter_(1, indices, True)

            subset[name].weight.data[weight_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print("Starting ...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    hf_device_map = _get_hf_device_map(model)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, dev, args.nsamples
        )

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in hf_device_map:
            dev = hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps = _move_to_device(inps, dev)
            outs = _move_to_device(outs, dev)
            attention_mask = _move_to_device(attention_mask, dev)
            position_ids = _move_to_device(position_ids, dev)
            position_embeddings = _move_to_device(position_embeddings, dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        layer_kwargs = _get_layer_kwargs(attention_mask, position_ids, position_embeddings)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        for handle in handles:
            handle.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print("Starting ...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    hf_device_map = _get_hf_device_map(model)

    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, dev, args.nsamples
        )

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in hf_device_map:
            dev = hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps = _move_to_device(inps, dev)
            outs = _move_to_device(outs, dev)
            attention_mask = _move_to_device(attention_mask, dev)
            position_ids = _move_to_device(position_ids, dev)
            position_embeddings = _move_to_device(position_embeddings, dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        layer_kwargs = _get_layer_kwargs(attention_mask, position_ids, position_embeddings)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        for handle in handles:
            handle.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            else:
                prune_mask = None

            gpts[name].fasterprune(
                args,
                args.sparsity_ratio,
                mask=prune_mask,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[i] = layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
