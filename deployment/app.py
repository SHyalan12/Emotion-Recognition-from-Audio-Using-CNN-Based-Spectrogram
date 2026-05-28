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
# ImageDataGenerator.flow_from_directory sorts classes alphabetically,
# so this order matches the training class indices exactly.
EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
EMOJIS = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲",
}

MODEL_PATHS = {
    "speech": "models/speech_model.h5",
    "song":   "models/song_model.h5",
}

# ── Your CNN models ───────────────────────────────────────────────────────────
my_models: dict = {}

def load_my_models():
    for mode, path in MODEL_PATHS.items():
        if os.path.exists(path):
            my_models[mode] = tf.keras.models.load_model(path)
            print(f"✓ Loaded {mode} model from {path}")
        else:
            print(f"⚠  {path} not found — upload your .h5 file to the models/ folder.")

load_my_models()

# ── Pre-trained baseline (lazy load — heavy, only when user picks it) ─────────
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

# ── Spectrogram pipeline (matches training exactly) ───────────────────────────
def audio_to_spectrogram_image(audio_path: str) -> Image.Image:
    """
    Replicates Lecture_recording_processing.ipynb:
      librosa.load → melspectrogram → power_to_db → matplotlib (3×3, axis off)
      → PNG → PIL resize to 128×128 RGB
    """
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

# ── Inference: your CNN ───────────────────────────────────────────────────────
def predict_my_model(audio_path: str, mode: str):
    if mode not in my_models:
        missing = MODEL_PATHS[mode]
        return None, f"⚠️ **{missing}** not found. Upload it to the `models/` folder."

    img = audio_to_spectrogram_image(audio_path)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    probs = my_models[mode].predict(arr, verbose=0)[0]

    top_i = int(np.argmax(probs))
    top = EMOTIONS[top_i]
    label_map = {f"{EMOJIS[e]} {e.capitalize()}": float(p) for e, p in zip(EMOTIONS, probs)}
    headline = f"### {EMOJIS[top]} {top.capitalize()} — {float(probs[top_i]):.1%}"
    return label_map, headline

# ── Inference: pre-trained baseline ───────────────────────────────────────────
def predict_pretrained(audio_path: str):
    pipe = get_pretrained()
    results = pipe(audio_path, top_k=8)   # list of {label, score}
    # Normalize label names to lowercase to match our EMOJIS map
    label_map = {}
    for r in results:
        name = r["label"].lower()
        emoji = EMOJIS.get(name, "🎵")
        label_map[f"{emoji} {name.capitalize()}"] = float(r["score"])

    top = max(results, key=lambda r: r["score"])
    name = top["label"].lower()
    emoji = EMOJIS.get(name, "🎵")
    headline = f"### {emoji} {name.capitalize()} — {top['score']:.1%}"
    return label_map, headline

# ── Top-level dispatcher ──────────────────────────────────────────────────────
def run(audio_path, model_choice):
    if audio_path is None:
        return None, "Please record or upload audio first."

    if model_choice == "My CNN — Speech":
        return predict_my_model(audio_path, "speech")
    if model_choice == "My CNN — Song":
        return predict_my_model(audio_path, "song")
    if model_choice == "Pre-trained (Wav2Vec2)":
        return predict_pretrained(audio_path)

    return None, "Unknown model selection."

# ── Compare-all mode ──────────────────────────────────────────────────────────
def run_all(audio_path):
    if audio_path is None:
        return None, None, None
    speech_chart, _ = predict_my_model(audio_path, "speech") if "speech" in my_models else (None, None)
    song_chart, _   = predict_my_model(audio_path, "song")   if "song"   in my_models else (None, None)
    pre_chart, _    = predict_pretrained(audio_path)
    return speech_chart, song_chart, pre_chart

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
                    value="My CNN — Speech",
                    label="Model",
                )
                run_btn = gr.Button("Detect Emotion 🔍", variant="primary")
            with gr.Column():
                headline_out = gr.Markdown()
                chart_out = gr.Label(num_top_classes=8, label="Confidence scores")

        run_btn.click(run, [audio_in, model_radio], [chart_out, headline_out])

    with gr.Tab("Compare all"):
        gr.Markdown("Run all three models on the same audio and compare side-by-side.")
        audio_in_2 = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        run_all_btn = gr.Button("Run All Models 🆚", variant="primary")
        with gr.Row():
            speech_chart = gr.Label(num_top_classes=8, label="My CNN — Speech")
            song_chart   = gr.Label(num_top_classes=8, label="My CNN — Song")
            pre_chart    = gr.Label(num_top_classes=8, label="Pre-trained Wav2Vec2")
        run_all_btn.click(run_all, [audio_in_2], [speech_chart, song_chart, pre_chart])

    gr.Markdown(
        "---\n"
        "**My CNN:** trained on mel spectrograms (128 × 128) — Speech 63 % · Song 78 %  \n"
        "**Baseline:** `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (fine-tuned on RAVDESS)  \n"
        "**Emotions:** angry · calm · disgust · fearful · happy · neutral · sad · surprised"
    )

if __name__ == "__main__":
    demo.launch()
