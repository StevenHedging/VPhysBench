from __future__ import annotations


class SceneAnalysisError(RuntimeError):
    """A stable, reportable failure raised by a scene observation pipeline."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ReferenceAnalysisError(SceneAnalysisError):
    """A reference-side observation defect that must not penalize a Baseline."""
