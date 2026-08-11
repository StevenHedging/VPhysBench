from importlib import import_module

from .task_compiler import build_task_instance, compile_task_instance

# Compatibility guard only: this name now fails closed so older callers cannot
# overwrite a schema-v2 AtomicRun. New code uses reevaluate_atomic_variant.
__all__ = [
    "build_task_instance",
    "compile_task_instance",
    "reevaluate_atomic",
    "reevaluate_atomic_variant",
    "run_atomic",
    "run_matrix",
]


_RUNTIME_EXPORTS = {
    "run_atomic": (".atomic_runner", "run_atomic"),
    "run_matrix": (".atomic_runner", "run_matrix"),
    "reevaluate_atomic": (".atomic_runner", "reevaluate_atomic"),
    "reevaluate_atomic_variant": (
        ".evaluation_variants",
        "reevaluate_atomic_variant",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _RUNTIME_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    return getattr(import_module(module_name, __name__), attribute_name)
