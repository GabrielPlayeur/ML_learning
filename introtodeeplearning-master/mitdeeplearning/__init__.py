import importlib

import mitdeeplearning.util  # noqa: F401  — cheap, and every lab uses it

# The lab modules load on first access rather than eagerly, because their imports are
# not a shared baseline: lab3 pulls openai + datasets, lab3_old pulls gym (unmaintained,
# no Python 3.12 wheel). Importing `mitdeeplearning` to use lab1 should not require them.
_SUBMODULES = ("lab1", "lab2", "lab3", "lab3_old")


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_SUBMODULES])
