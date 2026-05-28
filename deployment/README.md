# Deployment — Emotion Recognition Web App

Gradio-based web app that wraps the trained CNN models from this repo into a deployable demo.

## 🚀 Live demo
👉 **https://huggingface.co/spaces/shhyalan/emotion-recognition**

## What it does
Two modes side-by-side:
1. **My CNN models** — the speech and song models trained in `../Final_Model/`
2. **Pre-trained baseline** — `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` for comparison

Visitors can upload audio or record from their mic, pick a model, and see emotion predictions with confidence scores.

## Architecture
```
audio file
  └─► librosa.load()
        └─► melspectrogram → power_to_db
              └─► matplotlib figure (3×3, axis off) → PNG
                    └─► PIL resize (128×128) → normalize (÷255)
                          └─► CNN → softmax → emotion label
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Gradio app — UI + inference pipeline |
| `requirements.txt` | Python dependencies |
| `save_models_colab.py` | Snippet to paste at the end of training notebooks to export `.h5` |
| `models/` | Holds `speech_model.h5` and `song_model.h5` (not in git — too large) |

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Opens at `http://localhost:7860`.

## Deploy your own
1. Train the models in `../Final_Model/Final_model_speech.ipynb` and `../Final_Model/Final_model_song.ipynb`
2. Use `save_models_colab.py` to export `.h5` files
3. Push this folder to a HuggingFace Space (SDK: Gradio)
4. Upload `.h5` files to the Space's `models/` folder via HF web UI
