# scripts/generate_phoneme_tts.py
import os
import json
import time
from pathlib import Path

import pyttsx3
import soundfile as sf
import numpy as np
import torch
import torchaudio

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "reference_audio" / "phonemes"
TMP_DIR = ROOT / "data" / "reference_audio" / "phonemes_tmp"
META_FILE = ROOT / "metadata" / "phoneme_audio_index.json"

TARGET_SR = 16000
DEVICE = "cpu"

phoneme_examples = {
 "AA1":"bought","AE1":"cat","AH0":"about","AH1":"comma","AO1":"saw","AW1":"house",
 "AY1":"like","B":"bat","CH":"church","D":"dog","DH":"the","EH0":"taken","EH1":"bed",
 "ER0":"father","ER1":"bird","F":"fan","G":"go","HH":"hat","IH0":"cousin","IH1":"sit",
 "IY0":"happy","IY1":"she","JH":"judge","K":"cat","L":"let","M":"man","N":"no",
 "NG":"sing","OY1":"boy","P":"pen","R":"red","S":"sit","SH":"she","T":"top",
 "TH":"think","UH1":"put","UW1":"you","V":"van","W":"we","Y":"yes","Z":"zoo","ZH":"measure"
}

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)

def synthesize_with_pyttsx3(text, out_path):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 20)
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    time.sleep(0.2)

def save_audio_safe(path, waveform, sr):
    data = waveform.squeeze(0).numpy()
    sf.write(str(path), data, sr, subtype="PCM_16")

def resample_to_target(in_path, out_path, target_sr=TARGET_SR):
    data, sr = sf.read(str(in_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).unsqueeze(0)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    maxv = waveform.abs().max()
    if maxv > 0:
        waveform = waveform / maxv * 0.98

    save_audio_safe(out_path, waveform, target_sr)

def main():
    ensure_dirs()
    metadata = {}

    print("Starting phoneme TTS generation... total phonemes:", len(phoneme_examples))

    for phoneme, word in phoneme_examples.items():
        tmp_file = TMP_DIR / f"{phoneme}_tmp.wav"
        out_file = OUT_DIR / f"{phoneme}.wav"

        print(f"Synthesizing {phoneme} -> example word: '{word}'")
        synthesize_with_pyttsx3(word, tmp_file)
        resample_to_target(tmp_file, out_file, TARGET_SR)

        # NEW - use soundfile to read duration (avoids torchcodec)
        data, sr = sf.read(str(out_file), dtype="float32")
        if data.ndim > 1:
            length_frames = data.shape[0]
        else:
            length_frames = data.shape[0]
        duration = length_frames / sr

        metadata[phoneme] = {
            "file": str(out_file.relative_to(ROOT)),
            "example_word": word,
            "duration_s": round(duration, 3),
            "sr": TARGET_SR
        }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDONE! Generated phoneme audio in:", OUT_DIR)
    print("Metadata saved to:", META_FILE)

if __name__ == "__main__":
    main()
