from .atomic_runner import (
    build_task_instance,
    reevaluate_atomic,
    run_atomic,
    run_matrix,
)
from .evaluation_variants import reevaluate_atomic_variant

# Compatibility guard only: this name now fails closed so older callers cannot
# overwrite a schema-v2 AtomicRun. New code uses reevaluate_atomic_variant.
__all__ = [
    "build_task_instance",
    "reevaluate_atomic",
    "reevaluate_atomic_variant",
    "run_atomic",
    "run_matrix",
]
