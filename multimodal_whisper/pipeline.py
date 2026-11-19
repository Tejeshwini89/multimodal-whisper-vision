"""Orchestrate speech transcription, intent routing and visual inference."""
from __future__ import annotations

import re
from typing import Any

from .speech import WhisperTranscriber
from .vision import VisionEngine


_DESCRIBE_PATTERNS = (
    r"\bdescribe\b", r"\bcaption\b", r"\bwhat do you see\b",
    r"\btell me about (the )?(image|picture|photo)\b",
    r"\bexplain (the )?(image|picture|photo)\b",
)
_QUESTION_START = re.compile(
    r"^(?:what|who|where|when|why|how|which|whose|whom|is|are|am|was|were|do|does|did|can|could|would|will|has|have|had)\b",
    re.IGNORECASE,
)
_ACTION_START = re.compile(
    r"^(?:count|identify|find|locate|name|list|compare|read|tell|recognize|point out)\b",
    re.IGNORECASE,
)


def detect_intent(text: str) -> str:
    """Classify speech as a visual question, image description, or statement."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return "statement"
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in _DESCRIBE_PATTERNS):
        return "describe"
    if normalized.endswith("?") or _QUESTION_START.search(normalized) or _ACTION_START.search(normalized):
        return "question"
    return "statement"


class MultimodalPipeline:
    """Speech → Whisper → intent → BLIP/CLIP → structured result."""

    def __init__(
        self,
        whisper_model: str = "openai/whisper-small",
        device: str | None = None,
    ) -> None:
        self.speech = WhisperTranscriber(model_name=whisper_model, device=device)
        self.vision = VisionEngine(device=device)

    def run(
        self,
        audio_path: str,
        image: Any,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict[str, Any]:
        if image is None:
            raise ValueError("Please provide an image")
        asr = self.speech.transcribe(audio_path, language=language, task=task)
        transcript = asr["text"]
        if not transcript:
            raise RuntimeError("Whisper returned an empty transcript. Try a clearer or longer recording.")

        intent = detect_intent(transcript)
        caption = self.vision.caption(image)
        if intent == "question":
            answer = self.vision.answer(image, transcript)
        elif intent == "describe":
            answer = caption
        else:
            answer = f"I heard: {transcript}\nImage description: {caption}"

        similarity = self.vision.similarity(image, transcript)
        return {
            "transcript": transcript,
            "segments": asr["segments"],
            "task": asr["task"],
            "intent": intent,
            "image_caption": caption,
            "answer": answer,
            "speech_image_similarity": similarity,
            "models": {
                "speech": asr["model"],
                "captioning": self.vision.CAPTION_MODEL,
                "visual_question_answering": self.vision.VQA_MODEL,
                "image_text_similarity": self.vision.CLIP_MODEL,
            },
        }
