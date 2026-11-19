"""Multimodal Whisper + Vision."""
from .pipeline import MultimodalPipeline, detect_intent
from .speech import WhisperTranscriber
from .vision import VisionEngine

__all__ = ["MultimodalPipeline", "WhisperTranscriber", "VisionEngine", "detect_intent"]
__version__ = "1.0.0"
