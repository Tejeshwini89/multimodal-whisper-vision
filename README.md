# Multimodal Whisper + Vision

**Speak about an image.** This project combines Whisper speech recognition with BLIP image captioning and visual question answering, then uses CLIP to measure how closely the spoken text matches the supplied image.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tejeshwini89/multimodal-whisper-vision/blob/main/notebooks/colab_demo.ipynb)

## What it does

1. **Whisper ASR** converts an audio recording into text. Use `transcribe` to retain the spoken language or `translate` to produce English text.
2. **Intent routing** detects whether the speaker asks a visual question, requests a description, or makes a statement.
3. **BLIP captioning** generates a general description of the image.
4. **BLIP VQA** answers the transcribed question using the image.
5. **CLIP** computes cosine similarity between the image and the transcript as a lightweight grounding signal.

## Architecture

```text
Audio file
   │
   ▼
Whisper (ASR / speech translation)
   │
   ▼
Transcript ──► Intent detection ──┬── visual question ──► BLIP VQA ──┐
                                  ├── describe image ──► BLIP caption ├──► Result
                                  └── statement ───────► caption + text┘

Image ───────────────────────────────► BLIP caption / BLIP VQA
  │
  └──────────────────────────────────► CLIP image-text cosine similarity
```

## Run it in Google Colab (recommended)

1. Open [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** when available. CPU is supported but slower.
3. Run the notebook from top to bottom.
4. Upload one audio file and one image when prompted.
5. Read the transcript, detected intent, image caption, visual answer and CLIP similarity.

The notebook is self-contained: it does **not** require cloning the repository before running. It installs its dependencies, writes the package modules into the Colab runtime and downloads the pretrained models on first use.

## Supported input

- Audio: WAV, MP3, M4A, FLAC, OGG, WEBM and other formats that FFmpeg can decode.
- Image: common formats supported by Pillow, including PNG, JPEG and WEBP.
- Whisper tasks: `transcribe` or `translate` (speech translation to English).

## Model components

| Stage | Model | Purpose |
|---|---|---|
| Speech recognition | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | Multilingual speech-to-text and translation |
| Image captioning | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) | General image description |
| Visual question answering | [`Salesforce/blip-vqa-base`](https://huggingface.co/Salesforce/blip-vqa-base) | Answers a natural-language question about an image |
| Image-text similarity | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | Compares the transcript and image embeddings |

The BLIP model cards currently document direct model-class usage and note that the older BLIP task pipelines are not supported in Transformers 5. The Colab notebook therefore installs `transformers` 4.x and uses the model classes directly.

## Project layout

```text
multimodal-whisper-vision/
├── multimodal_whisper/
│   ├── __init__.py
│   ├── speech.py
│   ├── vision.py
│   └── pipeline.py
├── notebooks/
│   └── colab_demo.ipynb
├── tests/
│   └── test_intent.py
├── app.py                 # optional local Gradio UI
├── cli.py                 # optional command-line interface
├── requirements.txt
├── LICENSE
└── README.md
```

## Optional: run locally

Python 3.10+ and FFmpeg are required. A CUDA GPU is recommended.

```bash
# Install FFmpeg first (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install "gradio>=4.44"     # only if you want the optional web UI

python cli.py --audio question.mp3 --image photo.jpg
python app.py                   # optional Gradio UI
pytest -q
```

## Example output

```json
{
  "transcript": "How many cats are in the picture?",
  "intent": "question",
  "image_caption": "two cats sitting on a couch",
  "answer": "2",
  "speech_image_similarity": {
    "cosine_similarity": 0.27,
    "scaled_0_to_1": 0.635
  }
}
```

The values above are illustrative, not benchmark results. CLIP cosine similarity is a relative compatibility signal, **not a calibrated probability or proof that an answer is correct**. BLIP answers can also be wrong; verify important outputs against the image.

## Limitations and responsible use

- Whisper transcription quality varies with noise, accents, recording quality and language.
- BLIP VQA is a general-purpose model and may miscount objects, miss small details or hallucinate.
- CLIP similarity is not an answer-confidence score and should not be treated as one.
- Do not use model outputs as the sole basis for medical, legal, safety-critical or other high-impact decisions.
- Obtain consent before transcribing recordings of other people.
- Review each pretrained model's card and license before redistribution or commercial deployment. The component models have their own terms, separate from this repository's MIT license.

## Tests

The intent router can be tested without downloading neural network weights:

```bash
pytest -q
```

## Acknowledgements

This project uses pretrained models published by OpenAI and Salesforce AI Research through Hugging Face Transformers. See the linked model cards for original papers, training details, limitations and model-specific terms.
