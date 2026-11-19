"""BLIP image understanding and CLIP image-text similarity."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


class VisionEngine:
    """Load BLIP captioning, BLIP VQA and CLIP models once for repeated use."""

    CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
    VQA_MODEL = "Salesforce/blip-vqa-base"
    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        try:
            from transformers import (
                BlipForConditionalGeneration, BlipForQuestionAnswering,
                BlipProcessor, CLIPModel, CLIPProcessor,
            )
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with `pip install -r requirements.txt`.") from exc

        self.caption_processor = BlipProcessor.from_pretrained(self.CAPTION_MODEL)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            self.CAPTION_MODEL, torch_dtype=self.torch_dtype
        ).to(self.device).eval()

        self.vqa_processor = BlipProcessor.from_pretrained(self.VQA_MODEL)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(
            self.VQA_MODEL, torch_dtype=self.torch_dtype
        ).to(self.device).eval()

        self.clip_processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(
            self.CLIP_MODEL, torch_dtype=self.torch_dtype
        ).to(self.device).eval()

    @staticmethod
    def load_image(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as opened:
            return opened.convert("RGB")

    def _move_inputs(self, inputs: Any) -> Any:
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch_dtype)
        return inputs

    @torch.inference_mode()
    def caption(self, image: str | Path | Image.Image, max_new_tokens: int = 40) -> str:
        pil_image = self.load_image(image)
        inputs = self._move_inputs(self.caption_processor(images=pil_image, return_tensors="pt"))
        generated = self.caption_model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=3
        )
        return self.caption_processor.decode(generated[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def answer(self, image: str | Path | Image.Image, question: str, max_new_tokens: int = 24) -> str:
        if not question.strip():
            raise ValueError("The visual question cannot be empty")
        pil_image = self.load_image(image)
        inputs = self._move_inputs(
            self.vqa_processor(images=pil_image, text=question.strip(), return_tensors="pt")
        )
        generated = self.vqa_model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=2
        )
        return self.vqa_processor.decode(generated[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def similarity(self, image: str | Path | Image.Image, text: str) -> dict[str, float]:
        """Return cosine similarity; this is a ranking signal, not a probability."""
        pil_image = self.load_image(image)
        inputs = self._move_inputs(
            self.clip_processor(text=[text], images=pil_image, return_tensors="pt", padding=True, truncation=True)
        )
        outputs = self.clip_model(**inputs)
        image_embedding = F.normalize(outputs.image_embeds, dim=-1)
        text_embedding = F.normalize(outputs.text_embeds, dim=-1)
        cosine = F.cosine_similarity(image_embedding, text_embedding, dim=-1).item()
        return {
            "cosine_similarity": round(float(cosine), 4),
            "scaled_0_to_1": round(float((cosine + 1.0) / 2.0), 4),
        }
