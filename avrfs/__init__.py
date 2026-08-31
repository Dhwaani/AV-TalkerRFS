"""avrfs -- audio-visual talker tracking with modeled, mode-gated fusion.

The microphone array misses silence; the camera misses occlusion. These are
dual, physically-modeled failure modes, and this package puts both inside one
random-finite-set recursion instead of learning a fused embedding.
"""

from .activity import ActivityParams
from .filters import AVFilter, FilterConfig, FilterOutput, make_filter
from .metrics import TrackingScore, evaluate, ospa
from .sensors import AudioParams, VideoParams, sense_audio, sense_video
from .world import (CameraParams, OcclusionParams, World, WorldConfig,
                    make_world)

__version__ = "0.1.0"

__all__ = [
    "ActivityParams", "OcclusionParams", "CameraParams",
    "World", "WorldConfig", "make_world",
    "AudioParams", "VideoParams", "sense_audio", "sense_video",
    "AVFilter", "FilterConfig", "FilterOutput", "make_filter",
    "TrackingScore", "evaluate", "ospa",
    "__version__",
]
