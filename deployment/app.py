# ── Patch gradio_client schema bug (MUST run before importing gradio) ────────
# Bug: when JSON schemas have `additionalProperties: true/false` (bool instead
# of dict), gradio_client.utils crashes with "argument of type 'bool' is not
# iterable" during get_api_info(). This wraps the affected functions to handle
# non-dict schemas gracefully.
import gradio_client.utils as _gcu

if hasattr(_gcu, "_json_schema_to_python_type"):
    _orig_json = _gcu._json_schema_to_python_type
    def _safe_json_schema(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_json(schema, defs)
    _gcu._json_schema_to_python_type = _safe_json_schema

if hasattr(_gcu, "get_type"):
    _orig_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    _gcu.get_type = _safe_get_type
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import io
import os

# ── Labels ───────────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
EMOJIS = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲",
}

MODEL_PATHS = {
    "speech": "models/speech_model.h5",
    "song":   "models/song_model.h5",
}

# ── Custom CNN models ─────────────────────────────────────────────────────────
my_models: dict = {}

def load_my_models():
    for mode, path in MODEL_PATHS.items():
        if os.path.exists(path):
            my_models[mode] = tf.keras.models.load_model(path)
            print(f"✓ Loaded {mode} model from {path}")
        else:
            print(f"⚠  {path} not found — upload your .h5 file to the models/ folder.")

load_my_models()

# ── Pre-trained baseline (lazy-loaded — large model) ──────────────────────────
_pretrained_pipe = None

def get_pretrained():
    global _pretrained_pipe
    if _pretrained_pipe is None:
        print("Loading pre-trained Wav2Vec2 model (first run only)...")
        from transformers import pipeline
        _pretrained_pipe = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        )
        print("✓ Pre-trained model ready.")
    return _pretrained_pipe

# ── Spectrogram pipeline ──────────────────────────────────────────────────────
def audio_to_spectrogram_image(audio_path: str) -> Image.Image:
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((128, 128))

# ── HTML bar chart renderer (avoids gr.Label schema bug) ──────────────────────
def render_bar_chart(label_map: dict) -> str:
    """Build a simple HTML bar chart of confidence scores."""
    if not label_map:
        return "<p style='color:#888'>No data</p>"
    items = sorted(label_map.items(), key=lambda x: -x[1])
    rows = []
    for label, conf in items:
        pct = max(0.5, conf * 100)  # min width so 0% bars are still visible
        rows.append(
            f"<div style='margin:6px 0;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:14px;margin-bottom:2px;'>"
            f"<span>{label}</span><span style='color:#888;'>{conf*100:.1f}%</span>"
            f"</div>"
            f"<div style='background:#1f2937;border-radius:4px;height:8px;overflow:hidden;'>"
            f"<div style='background:linear-gradient(90deg,#8b5cf6,#6366f1);width:{pct}%;height:100%;'></div>"
            f"</div></div>"
        )
    return f"<div style='font-family:sans-serif;padding:8px;'>{''.join(rows)}</div>"

# ── Inference: custom CNN ─────────────────────────────────────────────────────
def predict_my_model(audio_path: str, mode: str):
    if mode not in my_models:
        missing = MODEL_PATHS[mode]
        return f"<p style='color:#fca5a5;padding:12px;'>⚠️ <b>{missing}</b> not found.<br>Upload your trained model to the <code>models/</code> folder.</p>", ""

    img = audio_to_spectrogram_image(audio_path)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs = my_models[mode].predict(arr, verbose=0)[0]

    top_i = int(np.argmax(probs))
    top = EMOTIONS[top_i]
    label_map = {f"{EMOJIS[e]} {e.capitalize()}": float(p) for e, p in zip(EMOTIONS, probs)}
    headline = f"### {EMOJIS[top]} {top.capitalize()} — {float(probs[top_i]):.1%}"
    return render_bar_chart(label_map), headline

# ── Inference: pre-trained baseline ───────────────────────────────────────────
def predict_pretrained(audio_path: str):
    pipe = get_pretrained()

    # Explicitly resample to 16 kHz (Wav2Vec2's training rate).
    # The HF pipeline sometimes skips auto-resampling, leading to flat/random output.
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    print(f"[pretrained] audio: {len(audio)} samples @ 16kHz "
          f"({len(audio)/16000:.1f}s), max amp: {np.abs(audio).max():.3f}")

    if len(audio) < 8000:  # less than 0.5s
        return ("<p style='color:#fbbf24;padding:12px;'>⚠️ Audio is too short. "
                "Please record at least 2 seconds.</p>", "")
    if np.abs(audio).max() < 0.01:
        return ("<p style='color:#fbbf24;padding:12px;'>⚠️ Audio is silent or too quiet. "
                "Please speak louder or check your mic.</p>", "")

    # Normalize amplitude to help the model
    audio = audio / (np.abs(audio).max() + 1e-9)

    # Pass as raw array with explicit sampling rate (bypasses pipeline's file decoding)
    results = pipe({"sampling_rate": 16000, "raw": audio}, top_k=8)

    label_map = {}
    for r in results:
        name = r["label"].lower()
        emoji = EMOJIS.get(name, "🎵")
        label_map[f"{emoji} {name.capitalize()}"] = float(r["score"])

    top = max(results, key=lambda r: r["score"])
    name = top["label"].lower()
    emoji = EMOJIS.get(name, "🎵")
    headline = f"### {emoji} {name.capitalize()} — {top['score']:.1%}"
    return render_bar_chart(label_map), headline

# ── Dispatcher ────────────────────────────────────────────────────────────────
def run(audio_path, model_choice):
    if audio_path is None:
        return "<p style='color:#fbbf24;padding:12px;'>Please record or upload audio first.</p>", ""

    if model_choice == "My CNN — Speech":
        return predict_my_model(audio_path, "speech")
    if model_choice == "My CNN — Song":
        return predict_my_model(audio_path, "song")
    if model_choice == "Pre-trained (Wav2Vec2)":
        return predict_pretrained(audio_path)
    return "<p>Unknown model.</p>", ""

def run_all(audio_path):
    if audio_path is None:
        empty = "<p style='color:#fbbf24;'>Please upload audio.</p>"
        return empty, empty, empty
    speech_html = predict_my_model(audio_path, "speech")[0] if "speech" in my_models else "<p style='color:#fca5a5;'>speech_model.h5 not uploaded</p>"
    song_html   = predict_my_model(audio_path, "song")[0]   if "song"   in my_models else "<p style='color:#fca5a5;'>song_model.h5 not uploaded</p>"
    pre_html    = predict_pretrained(audio_path)[0]
    return speech_html, song_html, pre_html

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Emotion Recognizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ Emotion Recognition from Audio\n"
        "Compare a custom CNN trained on mel spectrograms against a pre-trained Wav2Vec2 baseline."
    )

    with gr.Tab("Single model"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
                model_radio = gr.Radio(
                    choices=["My CNN — Speech", "My CNN — Song", "Pre-trained (Wav2Vec2)"],
                    value="Pre-trained (Wav2Vec2)",
                    label="Model",
                )
                run_btn = gr.Button("Detect Emotion 🔍", variant="primary")
            with gr.Column():
                headline_out = gr.Markdown()
                chart_out = gr.HTML(label="Confidence scores")

        run_btn.click(
            fn=run,
            inputs=[audio_in, model_radio],
            outputs=[chart_out, headline_out],
        )

    with gr.Tab("Compare all"):
        gr.Markdown("Run all three models on the same audio.")
        audio_in_2 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        run_all_btn = gr.Button("Run All Models 🆚", variant="primary")
        with gr.Row():
            speech_chart = gr.HTML(label="My CNN — Speech")
            song_chart   = gr.HTML(label="My CNN — Song")
            pre_chart    = gr.HTML(label="Pre-trained Wav2Vec2")
        run_all_btn.click(
            fn=run_all,
            inputs=[audio_in_2],
            outputs=[speech_chart, song_chart, pre_chart],
        )

    gr.Markdown(
        "---\n"
        "**My CNN:** trained on mel spectrograms (128 × 128) — Speech 63 % · Song 78 %  \n"
        "**Baseline:** `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (fine-tuned on RAVDESS)  \n"
        "**Emotions:** angry · calm · disgust · fearful · happy · neutral · sad · surprised"
    )

if __name__ == "__main__":
    demo.launch()
