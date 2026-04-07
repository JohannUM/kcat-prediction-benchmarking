from .model_wrapper.model import Model

def __getattr__(name):
    """Lazily load modules only when they are accessed."""
    if name == "plotting":
        import importlib
        return importlib.import_module(".plotting", __name__)
    elif name == "dataset":
        import importlib
        return importlib.import_module(".dataset", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'Model',
]