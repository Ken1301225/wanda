import importlib
import sys
import types
import unittest


class FakeCuda:
    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def device_count():
        return 0


class FakeRandom:
    @staticmethod
    def manual_seed(seed):
        return None


def fake_device(name):
    return name


def install_stub_dependencies():
    module_names = (
        "numpy",
        "torch",
        "torch.nn",
        "transformers",
        "datasets",
        "lib.sparsegpt",
        "lib.layerwrapper",
        "lib.data",
        "lib.ablate",
        "lib.prune",
        "lib.prune_qwen",
        "lib.prune_dsv2",
        "lib.eval",
        "main",
    )
    saved = {name: sys.modules.get(name) for name in module_names}

    numpy = types.ModuleType("numpy")
    numpy.random = FakeRandom()

    torch = types.ModuleType("torch")
    torch.device = fake_device
    torch.cuda = FakeCuda()
    torch.random = FakeRandom()
    torch.float16 = "float16"

    torch_nn = types.ModuleType("torch.nn")

    class Module:
        pass

    class Linear(Module):
        pass

    torch_nn.Module = Module
    torch_nn.Linear = Linear

    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise NotImplementedError

    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise NotImplementedError

    class Conv1D(Module):
        pass

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.Conv1D = Conv1D

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

    prune = types.ModuleType("lib.prune")
    prune_qwen = types.ModuleType("lib.prune_qwen")
    prune_dsv2 = types.ModuleType("lib.prune_dsv2")
    for module in (prune, prune_qwen, prune_dsv2):
        for name in ("prune_wanda", "prune_magnitude", "prune_sparsegpt", "prune_ablate", "check_sparsity", "find_layers"):
            setattr(module, name, lambda *args, **kwargs: None)

    eval_mod = types.ModuleType("lib.eval")
    eval_mod.eval_ppl = lambda *args, **kwargs: None
    eval_mod.eval_zero_shot = lambda *args, **kwargs: None

    sys.modules["numpy"] = numpy
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch_nn
    sys.modules["transformers"] = transformers
    sys.modules["datasets"] = datasets
    sys.modules["lib.sparsegpt"] = sparsegpt
    sys.modules["lib.layerwrapper"] = layerwrapper
    sys.modules["lib.data"] = data
    sys.modules["lib.ablate"] = ablate
    sys.modules["lib.prune"] = prune
    sys.modules["lib.prune_qwen"] = prune_qwen
    sys.modules["lib.prune_dsv2"] = prune_dsv2
    sys.modules["lib.eval"] = eval_mod

    sys.modules.pop("main", None)
    return saved


def restore_modules(saved):
    sys.modules.pop("main", None)
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class BackendRoutingTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = install_stub_dependencies()
        self.addCleanup(restore_modules, self.saved_modules)

    def test_main_routes_qwen_models_to_qwen_backend(self):
        main = importlib.import_module("main")

        qwen_backend = main.get_pruning_backend("Qwen/Qwen2.5-MoE-A2.7B")
        self.assertEqual(qwen_backend.__name__, "lib.prune_qwen")

    def test_main_routes_dsv2_models_to_dsv2_backend(self):
        main = importlib.import_module("main")

        dsv2_backend = main.get_pruning_backend("deepseek-ai/DeepSeek-V2-Lite")
        self.assertEqual(dsv2_backend.__name__, "lib.prune_dsv2")


if __name__ == "__main__":
    unittest.main()
