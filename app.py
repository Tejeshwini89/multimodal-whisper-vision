"""Optional local Gradio interface. Run with: python app.py"""
from __future__ import annotations

import gradio as gr

from multimodal_whisper import MultimodalPipeline

_PIPELINE = None


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = MultimodalPipeline()
    return _PIPELINE


def infer(audio_path, image, task):
    if audio_path is None or image is None:
        return "Upload both an audio recording and an image.", "", "", ""
    result = get_pipeline().run(audio_path, image, task=task)
    score = result["speech_image_similarity"]["cosine_similarity"]
    return (
        result["transcript"],
        result["image_caption"],
        result["answer"],
        f"Intent: {result['intent']} | CLIP cosine similarity: {score:.4f}",
    )


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Speak or upload audio"),
        gr.Image(type="pil", label="Image to understand"),
        gr.Radio(["transcribe", "translate"], value="transcribe", label="Whisper task"),
    ],
    outputs=[
        gr.Textbox(label="Whisper transcript"),
        gr.Textbox(label="BLIP image caption"),
        gr.Textbox(label="Visual answer"),
        gr.Textbox(label="Intent and CLIP similarity"),
    ],
    title="Multimodal Whisper + Vision",
    description="Speak about an image. Whisper transcribes your speech; BLIP understands the image; CLIP scores image-text similarity.",
)

if __name__ == "__main__":
    demo.launch()
