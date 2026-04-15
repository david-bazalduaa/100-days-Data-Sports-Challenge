"""Smoke tests — verify that the project structure and imports work correctly."""

import importlib


def test_src_package_imports():
    """Verify all src submodules are importable."""
    modules = [
        "src",
        "src.data",
        "src.features",
        "src.models",
        "src.viz",
        "src.validation",
        "src.dashboard",
        "src.scrapers",
    ]
    for module_name in modules:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


def test_version():
    """Verify package version is set."""
    import src

    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"
