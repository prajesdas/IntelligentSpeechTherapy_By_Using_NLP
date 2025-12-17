# scripts/generate_phoneme_tts.py
import json
import torch
import soundfile as sf
from pathlib import Path
import numpy as np
import warnings
import librosa

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "reference_audio" / "phonemes"
META_FILE = ROOT / "metadata" / "phoneme_audio_index.json"
TARGET_SR = 16000

# CMU Phoneme List
phoneme_examples = {
    "AA0": "odd", "AA1": "father", "AA2": "odd", "AE0": "at", "AE1": "cat", "AE2": "at",
    "AH0": "about", "AH1": "hut", "AH2": "about", "AO0": "off", "AO1": "dog", "AO2": "off",
    "AW0": "out", "AW1": "cow", "AW2": "out", "AY0": "hide", "AY1": "my", "AY2": "hide",
    "EH0": "ed", "EH1": "red", "EH2": "ed", "ER0": "letter", "ER1": "bird", "ER2": "letter",
    "EY0": "okay", "EY1": "say", "EY2": "okay", "IH0": "it", "IH1": "sit", "IH2": "it",
    "IY0": "happy", "IY1": "see", "IY2": "happy", "OW0": "go", "OW1": "go", "OW2": "go",
    "OY0": "toy", "OY1": "boy", "OY2": "toy", "UH0": "book", "UH1": "book", "UH2": "book",
    "UW0": "you", "UW1": "too", "UW2": "you",
    "B": "bad", "CH": "cheese", "D": "dog", "DH": "the", "F": "fan", "G": "go",
    "HH": "hat", "JH": "judge", "K": "cat", "L": "live", "M": "man", "N": "no",
    "NG": "sing", "P": "pen", "R": "red", "S": "sit", "SH": "she", "T": "top",
    "TH": "think", "V": "van", "W": "we", "Y": "yes", "Z": "zoo", "ZH": "measure"
}

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    
    # 1. Load Silero Model (Standard English Version)
    print("Loading Silero TTS (Open Source - English v3)...")
    device = torch.device('cpu')
    
    # --- FIX IS HERE: Use 'v3_en' instead of 'v3_en_indic' ---
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                         model='silero_tts',
                                         language='en',
                                         speaker='v3_en') 
    model.to(device)

    metadata = {}
    print(f"Generating {len(phoneme_examples)} phonemes...")

    for phoneme, word in phoneme_examples.items():
        out_file = OUT_DIR / f"{phoneme}.wav"
        
        # Generate Audio
        # 'en_0' to 'en_117' are available in v3_en. We use en_0 (Standard Male).
        try:
            audio = model.apply_tts(text=word,
                                    speaker='en_0',
                                    sample_rate=48000)
            
            # Convert Torch Tensor to Numpy
            audio_np = audio.numpy()
            
            # Resample to 16k (Required for Wav2Vec2)
            if len(audio_np) > 0:
                audio_16k = librosa.resample(audio_np, orig_sr=48000, target_sr=TARGET_SR)
                
                # Normalize Volume
                maxv = np.abs(audio_16k).max()
                if maxv > 0: audio_16k = audio_16k / maxv * 0.95

                # Save
                sf.write(str(out_file), audio_16k, TARGET_SR)
                
                duration = len(audio_16k) / TARGET_SR
                print(f"Generated {phoneme} -> {duration:.2f}s")

                metadata[phoneme] = {
                    "file": str(out_file.relative_to(ROOT)),
                    "example_word": word,
                    "duration_s": round(duration, 3),
                    "sr": TARGET_SR,
                }
        except Exception as e:
            print(f"Failed to generate {phoneme}: {e}")

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDONE! Open-Source Reference Audio Generated.")

if __name__ == "__main__":
    main()