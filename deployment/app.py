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
        print("Loading pre-trained SUPERB HuBERT model (first run only)...")
        from transformers import pipeline
        # Swapped from ehcalabres/wav2vec2 (returned ~uniform probs on real-world
        # audio) to SUPERB HuBERT, which is much more robust outside RAVDESS.
        # Trade-off: only 4 emotions (neu/hap/ang/sad) instead of 8.
        _pretrained_pipe = pipeline(
            "audio-classification",
            model="superb/hubert-large-superb-er",
        )
        print("✓ Pre-trained model ready.")
    return _pretrained_pipe

# Map SUPERB's short codes to our full-name + emoji UI labels
_SUPERB_LABEL_MAP = {
    "neu": ("neutral", "😐"),
    "hap": ("happy",   "😄"),
    "ang": ("angry",   "😠"),
    "sad": ("sad",     "😢"),
}

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

# ── Result rendering helpers ──────────────────────────────────────────────────
def render_headline(emoji: str, name: str, confidence: float) -> str:
    """Big result card: large emoji + emotion name + confidence percentage."""
    return (
        "<div style='text-align:center;padding:28px 16px;"
        "background:linear-gradient(135deg,#1e1b4b 0%,#312e81 60%,#4338ca 100%);"
        "border-radius:14px;border:1px solid rgba(139,92,246,0.25);'>"
        f"<div style='font-size:84px;line-height:1;'>{emoji}</div>"
        f"<div style='font-size:30px;font-weight:700;color:#f8fafc;margin-top:10px;letter-spacing:0.5px;'>{name.capitalize()}</div>"
        f"<div style='font-size:16px;color:#c7d2fe;margin-top:6px;'>{confidence:.1%} confidence</div>"
        "</div>"
    )

def render_compact_headline(emoji: str, name: str, confidence: float) -> str:
    """Smaller card for the side-by-side comparison tab."""
    return (
        "<div style='text-align:center;padding:16px 10px;"
        "background:linear-gradient(135deg,#1e1b4b,#312e81);"
        "border-radius:10px;border:1px solid rgba(139,92,246,0.2);'>"
        f"<div style='font-size:52px;line-height:1;'>{emoji}</div>"
        f"<div style='font-size:18px;font-weight:600;color:#f8fafc;margin-top:6px;'>{name.capitalize()}</div>"
        f"<div style='font-size:13px;color:#c7d2fe;margin-top:2px;'>{confidence:.1%}</div>"
        "</div>"
    )

def render_bar_chart(label_map: dict) -> str:
    """Full confidence distribution as horizontal bars (shown inside an accordion)."""
    if not label_map:
        return "<p style='color:#888'>No data</p>"
    items = sorted(label_map.items(), key=lambda x: -x[1])
    rows = []
    for label, conf in items:
        pct = max(0.5, conf * 100)
        rows.append(
            f"<div style='margin:8px 0;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:14px;margin-bottom:3px;'>"
            f"<span>{label}</span><span style='color:#9ca3af;'>{conf*100:.1f}%</span>"
            f"</div>"
            f"<div style='background:#1f2937;border-radius:4px;height:8px;overflow:hidden;'>"
            f"<div style='background:linear-gradient(90deg,#8b5cf6,#6366f1);width:{pct}%;height:100%;'></div>"
            f"</div></div>"
        )
    return f"<div style='font-family:sans-serif;padding:10px 4px;'>{''.join(rows)}</div>"

def warning_card(message: str) -> str:
    return (
        f"<div style='padding:18px;border-radius:10px;background:#422006;"
        f"border:1px solid #d97706;color:#fde68a;text-align:center;'>{message}</div>"
    )

# ── Inference: spectrogram CNN ────────────────────────────────────────────────
def predict_my_model(audio_path: str, mode: str, compact: bool = False):
    """Returns (headline_html, chart_html)."""
    if mode not in my_models:
        missing = MODEL_PATHS[mode]
        return warning_card(f"⚠️ Model not loaded: <code>{missing}</code>"), ""

    img = audio_to_spectrogram_image(audio_path)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs = my_models[mode].predict(arr, verbose=0)[0]

    top_i = int(np.argmax(probs))
    top_name = EMOTIONS[top_i]
    top_emoji = EMOJIS[top_name]
    top_conf = float(probs[top_i])

    label_map = {f"{EMOJIS[e]} {e.capitalize()}": float(p) for e, p in zip(EMOTIONS, probs)}
    headline_fn = render_compact_headline if compact else render_headline
    return headline_fn(top_emoji, top_name, top_conf), render_bar_chart(label_map)

# ── Inference: pre-trained baseline ───────────────────────────────────────────
def predict_pretrained(audio_path: str, compact: bool = False):
    """Returns (headline_html, chart_html)."""
    pipe = get_pretrained()

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    print(f"[pretrained] audio: {len(audio)} samples @ 16kHz "
          f"({len(audio)/16000:.1f}s), max amp: {np.abs(audio).max():.3f}")

    if len(audio) < 8000:
        return warning_card("⚠️ Audio too short — please record at least 2 seconds."), ""
    if np.abs(audio).max() < 0.01:
        return warning_card("⚠️ Audio is silent or too quiet — please speak louder."), ""

    audio = audio / (np.abs(audio).max() + 1e-9)
    results = pipe({"sampling_rate": 16000, "raw": audio}, top_k=4)

    label_map = {}
    for r in results:
        code = r["label"].lower()
        name, emoji = _SUPERB_LABEL_MAP.get(code, (code, "🎵"))
        label_map[f"{emoji} {name.capitalize()}"] = float(r["score"])

    top = max(results, key=lambda r: r["score"])
    top_name, top_emoji = _SUPERB_LABEL_MAP.get(top["label"].lower(), (top["label"], "🎵"))
    top_conf = float(top["score"])

    headline_fn = render_compact_headline if compact else render_headline
    return headline_fn(top_emoji, top_name, top_conf), render_bar_chart(label_map)

# ── Model display names ───────────────────────────────────────────────────────
MODEL_CHOICES = [
    "Spectrogram CNN — Speech",
    "Spectrogram CNN — Song",
    "HuBERT Baseline (Pretrained)",
]

# ── Dispatcher ────────────────────────────────────────────────────────────────
def run(audio_path, model_choice):
    """Returns (headline_html, chart_html) for the Single-model tab."""
    if audio_path is None:
        return warning_card("Please record or upload audio first."), ""

    if model_choice == "Spectrogram CNN — Speech":
        return predict_my_model(audio_path, "speech")
    if model_choice == "Spectrogram CNN — Song":
        return predict_my_model(audio_path, "song")
    if model_choice == "HuBERT Baseline (Pretrained)":
        return predict_pretrained(audio_path)
    return warning_card("Unknown model selected."), ""

def run_all(audio_path):
    """Returns 6 outputs: (headline, chart) × 3 models."""
    if audio_path is None:
        w = warning_card("Please record or upload audio.")
        return w, "", w, "", w, ""

    if "speech" in my_models:
        s_head, s_chart = predict_my_model(audio_path, "speech", compact=True)
    else:
        s_head, s_chart = warning_card("Speech model not loaded"), ""

    if "song" in my_models:
        g_head, g_chart = predict_my_model(audio_path, "song", compact=True)
    else:
        g_head, g_chart = warning_card("Song model not loaded"), ""

    p_head, p_chart = predict_pretrained(audio_path, compact=True)

    return s_head, s_chart, g_head, g_chart, p_head, p_chart

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Speech Emotion Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎙️ Speech Emotion Recognition\n"
        "A side-by-side comparison of a **custom mel-spectrogram CNN** and the "
        "**HuBERT (SUPERB) baseline** for audio emotion classification."
    )

    # ─── Single-model tab ────────────────────────────────────────────────────
    with gr.Tab("Analyze"):
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio input",
                )
                model_radio = gr.Radio(
                    choices=MODEL_CHOICES,
                    value="HuBERT Baseline (Pretrained)",
                    label="Model",
                )
                run_btn = gr.Button("Analyze Emotion", variant="primary", size="lg")

            with gr.Column(scale=1):
                headline_out = gr.HTML()
                with gr.Accordion("View full confidence distribution", open=False):
                    chart_out = gr.HTML()

        run_btn.click(
            fn=run,
            inputs=[audio_in, model_radio],
            outputs=[headline_out, chart_out],
        )

    # ─── Compare-all tab ─────────────────────────────────────────────────────
    with gr.Tab("Compare models"):
        gr.Markdown("Run all three models on the same audio clip and compare side-by-side.")
        audio_in_2 = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Audio input",
        )
        run_all_btn = gr.Button("Run All Models", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Spectrogram CNN — Speech")
                speech_head = gr.HTML()
                with gr.Accordion("Distribution", open=False):
                    speech_chart = gr.HTML()
            with gr.Column():
                gr.Markdown("### Spectrogram CNN — Song")
                song_head = gr.HTML()
                with gr.Accordion("Distribution", open=False):
                    song_chart = gr.HTML()
            with gr.Column():
                gr.Markdown("### HuBERT Baseline")
                pre_head = gr.HTML()
                with gr.Accordion("Distribution", open=False):
                    pre_chart = gr.HTML()

        run_all_btn.click(
            fn=run_all,
            inputs=[audio_in_2],
            outputs=[
                speech_head, speech_chart,
                song_head, song_chart,
                pre_head, pre_chart,
            ],
        )

    gr.Markdown(
        "---\n"
        "### About the models\n"
        "**Spectrogram CNN** — A convolutional network trained from scratch on "
        "128 × 128 mel-spectrogram images. Separate weights for speech (63 % val. accuracy) "
        "and song (78 % val. accuracy), each covering 8 emotion classes: "
        "*angry, calm, disgust, fearful, happy, neutral, sad, surprised*.  \n\n"
        "**HuBERT Baseline** — `superb/hubert-large-superb-er`, a transformer "
        "model fine-tuned on the SUPERB emotion recognition benchmark. Predicts 4 classes: "
        "*neutral, happy, angry, sad*.  \n\n"
        "*Tip: clear, expressive speech of 3 – 5 seconds gives the best results.*"
    )

if __name__ == "__main__":
    demo.launch()
