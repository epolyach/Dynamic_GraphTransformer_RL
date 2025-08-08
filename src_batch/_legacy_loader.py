import importlib.util
import sys
import os

def load_legacy_module(current_name: str, legacy_path: str):
    """Load a legacy Python module from a file path and bind it to current_name in sys.modules.
    This allows `import current_name` and `from current_name import X` to work transparently.
    """
    legacy_path = os.path.abspath(legacy_path)
    spec = importlib.util.spec_from_file_location(current_name, legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    sys.modules[current_name] = module
    return module

