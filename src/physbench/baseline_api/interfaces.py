from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..domain import (
    AtomicPlan,
    BaselineBundle,
    BaselineTaskInstance,
    DatasetSnapshot,
    TaskSpec,
)


class DataAdapter(ABC):
    """Baseline-owned conversion from an immutable Case to native model inputs.

    A concrete adapter owns the complete adaptation pipeline: spatial, temporal,
    input-paradigm, generic text, and optional physics injection.  The benchmark
    passes ``native_inputs`` through to the Baseline. Managed standard adapters
    use an auditable text/vision/shape envelope; advanced adapters may instead
    expose opaque tokens, tensors, control streams, or another model-native
    mechanism.
    """

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Fingerprint of the complete adaptation pipeline."""
        raise NotImplementedError

    @property
    @abstractmethod
    def materialization_fingerprint(self) -> str:
        """Fingerprint of stages that create reusable media derivatives."""
        raise NotImplementedError

    def dependency_paths(self) -> dict[str, Path]:
        """Return output-affecting shared files outside the Baseline Bundle.

        Bundle-local adapter code is automatically covered by the portable
        Bundle digest. Adapters only need to report shared implementation,
        profile, solver, or encoder files that live elsewhere.
        """
        return {}

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def adapt_case(
        self, case: dict[str, Any], *, role: str
    ) -> dict[str, Any]:
        """Return an auditable baseline-native adaptation record."""
        raise NotImplementedError


class TaskBuilder(ABC):
    """Compile a Benchmark-owned plan into one Baseline task instance.

    The Benchmark planner expands Dataset + Task before invoking this boundary.
    Builders are deterministic and side-effect free: they validate compatibility,
    invoke their private DataAdapter, assemble an operation graph and seal a
    runnable specification. They cannot select cases, train, or infer.
    """

    bundle: BaselineBundle
    data_adapter: DataAdapter

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def compile(
        self,
        dataset: DatasetSnapshot,
        task: TaskSpec,
        canonical_plan: AtomicPlan,
    ) -> BaselineTaskInstance:
        raise NotImplementedError


class BaselinePlugin(ABC):
    """Execute a sealed task instance produced by this Baseline's TaskBuilder."""

    bundle: BaselineBundle
    task_builder: TaskBuilder

    @abstractmethod
    def run_task(
        self,
        *,
        instance: BaselineTaskInstance,
        run_dir: Path,
        execute: bool,
        stop_after_training: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return the training stage and prediction records."""
        raise NotImplementedError
