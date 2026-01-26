# scripts/generate_phoneme_tts.py
import json
import os
import torch
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "reference_audio" / "phonemes"
META_FILE = ROOT / "metadata" / "phoneme_audio_index.json"

TARGET_SR = 16000

# Standard set of words containing the target phonemes
phoneme_examples = {
    "AA1": "bought", "AE1": "cat", "AH0": "about", "AH1": "comma",
    "AO1": "saw", "AW1": "house", "AY1": "like", "B": "bat",
    "CH": "church", "D": "dog", "DH": "the", "EH0": "taken",
    "EH1": "bed", "ER0": "father", "ER1": "bird", "F": "fan",
    "G": "go", "HH": "hat", "IH0": "cousin", "IH1": "sit",
    "IY0": "happy", "IY1": "she", "JH": "judge", "K": "cat",
    "L": "let", "M": "man", "N": "no", "NG": "sing",
    "OY1": "boy", "P": "pen", "R": "red", "S": "sit",
    "SH": "she", "T": "top", "TH": "think", "UH1": "put",
    "UW1": "you", "V": "van", "W": "we", "Y": "yes",
    "Z": "zoo", "ZH": "measure",
}

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_silero_model():
    """Load Silero TTS model from torch hub (downloads once)."""
    print("Loading local Silero TTS model...")
    device = torch.device('cpu')
    
    # This downloads the model to your torch cache folder (~100MB)
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker='v3_en')
    model.to(device)
    return model

def post_process_audio(audio_tensor, out_path, model_sr=48000, target_sr=TARGET_SR):
    try:
        # Convert PyTorch tensor to numpy
        data = audio_tensor.squeeze().cpu().numpy()
        
        # Resample from model rate (48k) to target rate (16k)
        if model_sr != target_sr:
            data = librosa.resample(data, orig_sr=model_sr, target_sr=target_sr)
        
        # Trim silence
        data, _ = librosa.effects.trim(data, top_db=30)
        
        # Normalize volume
        maxv = np.abs(data).max()
        if maxv > 0:
            data = data / maxv * 0.95
            
        sf.write(str(out_path), data, target_sr)
        return len(data) / target_sr
    except Exception as e:
        print(f"[WARN] Processing error for {out_path}: {e}")
        return 0.0

def main():
    ensure_dirs()
    model = load_silero_model()
    speaker = 'en_0' # Standard male voice
    sample_rate = 48000
    
    metadata = {}
    print(f"Starting Local Generation (Silero TTS)...")
    
    success_count = 0
    total = len(phoneme_examples)

    for i, (phoneme, word) in enumerate(phoneme_examples.items()):
        out_file = OUT_DIR / f"{phoneme}.wav"
        
        try:
            # Generate Audio Locally
            audio = model.apply_tts(text=word,
                                    speaker=speaker,
                                    sample_rate=sample_rate)
            
            # Process and Save
            duration = post_process_audio(audio, out_file, model_sr=sample_rate)
            
            print(f"[{i+1}/{total}] {phoneme} ({word}) -> {duration:.2f}s")
            
            metadata[phoneme] = {
                "file": str(out_file.relative_to(ROOT)),
                "example_word": word,
                "duration_s": round(duration, 3),
                "sr": TARGET_SR,
            }
            success_count += 1
            
        except Exception as e:
            print(f"[{i+1}/{total}] FAILED {phoneme}: {e}")

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDONE! Generated {success_count}/{total} phoneme files locally.")

if __name__ == "__main__":
    main()