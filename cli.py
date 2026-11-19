"""Command-line entry point. Example: python cli.py --audio question.wav --image photo.jpg"""
from __future__ import annotations

import argparse
import json

from multimodal_whisper import MultimodalPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about an image using your voice.")
    parser.add_argument("--audio", required=True, help="Path to an audio file")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--model", default="openai/whisper-small", help="Hugging Face Whisper model ID")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--language", default=None, help="Optional Whisper language hint, e.g. English")
    args = parser.parse_args()

    pipeline = MultimodalPipeline(whisper_model=args.model)
    result = pipeline.run(args.audio, args.image, language=args.language, task=args.task)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
