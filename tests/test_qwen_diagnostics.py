import importlib
import sys
import types
import unittest


def install_stub_dependencies():
    module_names = (
        "torch",
        "torch.nn",
        "transformers",
        "datasets",
        "lib.sparsegpt",
        "lib.layerwrapper",
        "lib.data",
        "lib.ablate",
        "lib.prune_qwen",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    torch.no_grad = lambda: (lambda fn: fn)

    torch_nn = types.ModuleType("torch.nn")

    class Module:
        pass

    class Linear(Module):
        pass

    torch_nn.Module = Module
    torch_nn.Linear = Linear

    transformers = types.ModuleType("transformers")
    transformers.Conv1D = type("Conv1D", (), {})

    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: None

    sparsegpt = types.ModuleType("lib.sparsegpt")
    sparsegpt.SparseGPT = object

    layerwrapper = types.ModuleType("lib.layerwrapper")
    layerwrapper.WrappedGPT = object

    data = types.ModuleType("lib.data")
    data.get_loaders = lambda *args, **kwargs: ([], None)

    ablate = types.ModuleType("lib.ablate")
    ablate.AblateGPT = object

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch_nn
    sys.modules["transformers"] = transformers
    sys.modules["datasets"] = datasets
    sys.modules["lib.sparsegpt"] = sparsegpt
    sys.modules["lib.layerwrapper"] = layerwrapper
    sys.modules["lib.data"] = data
    sys.modules["lib.ablate"] = ablate
    sys.modules.pop("lib.prune_qwen", None)

    return saved


def restore_modules(saved):
    sys.modules.pop("lib.prune_qwen", None)
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class FakeWrappedLayer:
    def __init__(self, nsamples, scaler_row):
        self.nsamples = nsamples
        self.scaler_row = scaler_row


class QwenDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = install_stub_dependencies()
        self.addCleanup(restore_modules, self.saved_modules)

    def test_summarize_expert_activity_reports_zero_hit_and_zero_columns(self):
        prune_qwen = importlib.import_module("lib.prune_qwen")
        wrapped_layers = {
            "layer.experts.0": FakeWrappedLayer(0, [0.0, 0.0, 0.0]),
            "layer.experts.1": FakeWrappedLayer(3, [1.0, 0.0, 2.0]),
        }

        summary = prune_qwen._summarize_expert_activity(wrapped_layers)

        self.assertEqual(summary["total_experts"], 2)
        self.assertEqual(summary["zero_hit_experts"], 1)
        self.assertEqual(summary["zero_scaler_columns"], 4)
        self.assertEqual(summary["min_nsamples"], 0)
        self.assertEqual(summary["max_nsamples"], 3)


if __name__ == "__main__":
    unittest.main()
