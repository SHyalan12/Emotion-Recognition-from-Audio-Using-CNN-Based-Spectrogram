# ─────────────────────────────────────────────────────────────────────────────
# Paste this cell at the END of both Final_model_speech.ipynb
# and Final_model_song.ipynb in Google Colab, then run it.
#
# It saves the trained model and downloads it straight to your computer.
# ─────────────────────────────────────────────────────────────────────────────

# In Final_model_speech.ipynb  →  save as speech_model.h5
model.save("speech_model.h5")
print("Saved speech_model.h5")

from google.colab import files
files.download("speech_model.h5")

# ── In Final_model_song.ipynb use this instead ────────────────────────────────
# model.save("song_model.h5")
# print("Saved song_model.h5")
# from google.colab import files
# files.download("song_model.h5")
