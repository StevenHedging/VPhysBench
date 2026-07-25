"""Reusable observation, geometry, fitting, and artifact primitives."""

from .errors import SceneAnalysisError
from .media import (
    SampledVideo,
    VideoInfo,
    VideoProtocolError,
    probe_video,
    sample_video,
)
from .similarity import (
    bounded_ratio_similarity,
    exponential_delta_similarity,
    relative_delta,
    scaled_delta,
)

__all__ = [
    "SampledVideo",
    "SceneAnalysisError",
    "VideoInfo",
    "VideoProtocolError",
    "probe_video",
    "sample_video",
    "bounded_ratio_similarity",
    "exponential_delta_similarity",
    "relative_delta",
    "scaled_delta",
]
