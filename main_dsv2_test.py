import argparse
import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def safe_version(pkg_name: str) -> str:
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return "not-installed"


def normalize_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    return path_value.rstrip("/")


def dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_dynamic_sources(model_path: str, cfg) -> dict:
    auto_map = getattr(cfg, "auto_map", {}) or {}
    result = {
        "auto_map": auto_map,
        "config_class_ref": auto_map.get("AutoConfig"),
        "model_class_ref": auto_map.get("AutoModelForCausalLM"),
        "config_module": None,
        "config_file": None,
        "model_module": None,
        "model_file": None,
    }

    config_ref = result["config_class_ref"]
    if config_ref:
        config_cls = get_class_from_dynamic_module(config_ref, model_path)
        config_module = config_cls.__module__
        result["config_module"] = config_module
        result["config_file"] = getattr(sys.modules.get(config_module), "__file__", None)

    model_ref = result["model_class_ref"]
    if model_ref:
        model_cls = get_class_from_dynamic_module(model_ref, model_path)
        model_module = model_cls.__module__
        result["model_module"] = model_module
        result["model_file"] = getattr(sys.modules.get(model_module), "__file__", None)

    return result


def inspect_weight_index(model_path: str) -> dict:
    index_path = Path(model_path) / "model.safetensors.index.json"
    summary = {
        "index_path": str(index_path),
        "index_exists": index_path.exists(),
        "num_keys": 0,
        "has_fused_weight": False,
        "has_mlp_experts": False,
    }
    if not index_path.exists():
        return summary

    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    keys = list(weight_map.keys())
    summary["num_keys"] = len(keys)
    summary["has_fused_weight"] = any("fused_weight" in k for k in keys)
    summary["has_mlp_experts"] = any(".mlp.experts." in k for k in keys)
    return summary


def copy_source_file(src_path: str | None, out_dir: Path) -> str | None:
    if not src_path:
        return None
    src = Path(src_path)
    if not src.exists():
        return None
    dst = out_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def print_env() -> None:
    print("torch", safe_version("torch"))
    print("transformers", safe_version("transformers"))
    print("accelerate", safe_version("accelerate"))
    print("# of gpus:", torch.cuda.device_count())


def run_probe(args) -> dict:
    cfg = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    source_info = resolve_dynamic_sources(args.model, cfg)
    index_info = inspect_weight_index(args.model)

    print("probe mode: source/keyspace only, skip model load/prune/eval")
    print(f"probe _name_or_path: {getattr(cfg, '_name_or_path', '<unknown>')}")
    print(f"probe auto_map AutoModelForCausalLM: {source_info['model_class_ref']}")
    print(f"probe class module: {source_info['model_module']}")
    print(f"probe class file: {source_info['model_file']}")
    print(
        "probe weight keys: "
        f"total={index_info['num_keys']}, "
        f"has_fused_weight={index_info['has_fused_weight']}, "
        f"has_mlp_experts={index_info['has_mlp_experts']}"
    )

    return {
        "mode": "probe",
        "model": args.model,
        "name_or_path": getattr(cfg, "_name_or_path", None),
        "source_info": source_info,
        "index_info": index_info,
    }


def run_quick_save(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    source_info = resolve_dynamic_sources(args.model, cfg)
    index_info = inspect_weight_index(args.model)

    cfg.save_pretrained(out_dir)
    copied_config_file = copy_source_file(source_info.get("config_file"), out_dir)
    copied_model_file = copy_source_file(source_info.get("model_file"), out_dir)

    tokenizer_saved = False
    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        tokenizer.save_pretrained(out_dir)
        tokenizer_saved = True

    model_saved = False
    if args.with_weights:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            low_cpu_mem_usage=True,
            device_map=args.device_map,
            torch_dtype=dtype_from_name(args.torch_dtype),
        )
        print(f"quick-save model module: {model.__class__.__module__}")
        print(
            "quick-save model file: "
            f"{getattr(sys.modules.get(model.__class__.__module__), '__file__', '<unknown>')}"
        )
        model.save_pretrained(
            out_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        model_saved = True

    print(f"quick-save output dir: {out_dir}")
    print(f"quick-save copied config file: {copied_config_file}")
    print(f"quick-save copied model file: {copied_model_file}")
    print(f"quick-save saved tokenizer: {tokenizer_saved}")
    print(f"quick-save saved weights: {model_saved}")

    return {
        "mode": "quick-save",
        "model": args.model,
        "out": str(out_dir),
        "name_or_path": getattr(cfg, "_name_or_path", None),
        "source_info": source_info,
        "index_info": index_info,
        "copied_config_file": copied_config_file,
        "copied_model_file": copied_model_file,
        "saved_tokenizer": tokenizer_saved,
        "saved_weights": model_saved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone probe and quick-save tester for DeepSeek/Wanda flows.")
    parser.add_argument("--mode", choices=["probe", "quick-save"], default="probe")
    parser.add_argument("--model", required=True, type=str, help="Model path or repo id")
    parser.add_argument("--out", type=str, default=None, help="Output dir for quick-save artifacts")
    parser.add_argument("--with-weights", action="store_true", help="Also load model and save weights (slower)")
    parser.add_argument("--save-tokenizer", action="store_true", help="Save tokenizer files in quick-save mode")
    parser.add_argument("--local-files-only", action="store_true", help="Disable remote fetches from HF")
    parser.add_argument("--device-map", type=str, default="cpu", help="device_map for --with-weights")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--report-name", type=str, default="quick_test_report.json")
    args = parser.parse_args()

    args.model = normalize_path(args.model)
    args.out = normalize_path(args.out)

    if args.mode == "quick-save" and not args.out:
        parser.error("--out is required when --mode quick-save")

    print_env()

    if args.mode == "probe":
        report = run_probe(args)
    else:
        report = run_quick_save(args)

    if args.out:
        report_path = Path(args.out) / args.report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"report saved: {report_path}")


if __name__ == "__main__":
    main()
