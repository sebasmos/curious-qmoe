#!/usr/bin/env python
"""Test all dependencies are correctly installed and working."""

import sys


def test_import(name, import_name=None, version_attr="__version__"):
    """Test if a module can be imported and optionally print version."""
    try:
        module = __import__(import_name or name)
        version = getattr(module, version_attr, None)
        version_str = f" ({version})" if version else ""
        print(f"  ✓ {name}{version_str}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: {e}")
        return False


def test_torch_backends():
    """Test PyTorch backend availability."""
    import torch

    print("\n  PyTorch backends:")
    print(f"    - CUDA available: {torch.cuda.is_available()}")
    print(f"    - MPS available: {torch.backends.mps.is_available()}")
    print(f"    - CPU: always available")
    return True


def main():
    print("=" * 50)
    print("QWave Dependency Test")
    print("=" * 50)

    results = []

    # Core ML
    print("\n[Core ML]")
    results.append(test_import("torch"))
    results.append(test_import("torchvision"))
    results.append(test_import("numpy"))
    results.append(test_import("scipy"))
    results.append(test_import("scikit-learn", "sklearn"))

    # Test torch backends
    if results[-5]:  # if torch imported successfully
        test_torch_backends()

    # Data
    print("\n[Data Processing]")
    results.append(test_import("pandas"))
    results.append(test_import("Pillow", "PIL", version_attr="__version__"))

    # Visualization
    print("\n[Visualization]")
    results.append(test_import("matplotlib"))
    results.append(test_import("seaborn"))

    # Config
    print("\n[Configuration]")
    results.append(test_import("hydra-core", "hydra"))
    results.append(test_import("omegaconf"))

    # Models
    print("\n[Models]")
    results.append(test_import("timm"))
    results.append(test_import("transformers"))
    results.append(test_import("clip"))

    # Utilities
    print("\n[Utilities]")
    results.append(test_import("tqdm"))
    results.append(test_import("tabulate"))
    results.append(test_import("joblib"))
    results.append(test_import("psutil"))
    results.append(test_import("sympy"))
    results.append(test_import("ftfy"))
    results.append(test_import("regex"))
    results.append(test_import("setuptools"))

    # Profiling
    print("\n[Profiling]")
    results.append(test_import("codecarbon"))
    results.append(test_import("memory_profiler"))
    results.append(test_import("fvcore"))

    # Summary
    passed = sum(results)
    total = len(results)
    failed = total - passed

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"All {total} dependencies installed correctly!")
    else:
        print(f"Result: {passed}/{total} passed, {failed} failed")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
