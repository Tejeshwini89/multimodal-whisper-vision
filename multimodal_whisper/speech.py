"""Whisper-based transcription and speech translation."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch


class WhisperTranscriber:
    """Transcribe or translate an audio file with a Hugging Face Whisper model.

    Audio is normalized to mono, 16 kHz WAV with ffmpeg before inference. This
    avoids relying on inconsistent input codecs or sample rates.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str | None = None,
        chunk_length_s: int = 30,
        stride_length_s: tuple[int, int] = (5, 5),
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.model_name = model_name

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with `pip install -r requirements.txt`.") from exc

        self.asr = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=0 if self.device.startswith("cuda") else -1,
            torch_dtype=self.torch_dtype,
            chunk_length_s=chunk_length_s,
        )

    @staticmethod
    def _normalize_audio(audio_path: str | Path) -> tuple[str, tempfile.TemporaryDirectory[str]]:
        source = Path(audio_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Audio file not found: {source}")

        temp_dir = tempfile.TemporaryDirectory(prefix="multimodal_whisper_")
        wav_path = str(Path(temp_dir.name) / "normalized.wav")
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            temp_dir.cleanup()
            raise RuntimeError("ffmpeg is required. Install it with `apt-get install ffmpeg`.") from exc
        except subprocess.CalledProcessError as exc:
            temp_dir.cleanup()
            details = (exc.stderr or "").strip()
            raise ValueError(f"Could not decode audio file {source.name}: {details}") from exc
        return wav_path, temp_dir

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict[str, Any]:
        """Return transcript and chunk timestamps. `task='translate'` outputs English."""
        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")

        normalized_path, temp_dir = self._normalize_audio(audio_path)
        generate_kwargs: dict[str, Any] = {"task": task}
        if language:
            generate_kwargs["language"] = language

        try:
            output = self.asr(
                normalized_path,
                return_timestamps=True,
                stride_length_s=self.stride_length_s,
                generate_kwargs=generate_kwargs,
            )
        finally:
            temp_dir.cleanup()

        chunks = output.get("chunks") or []
        segments = []
        for chunk in chunks:
            timestamp = chunk.get("timestamp")
            segments.append({
                "start": timestamp[0] if timestamp and len(timestamp) > 0 else None,
                "end": timestamp[1] if timestamp and len(timestamp) > 1 else None,
                "text": (chunk.get("text") or "").strip(),
            })

        return {
            "text": (output.get("text") or "").strip(),
            "segments": segments,
            "task": task,
            "language_hint": language,
            "model": self.model_name,
        }
