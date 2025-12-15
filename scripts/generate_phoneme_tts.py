# scripts/generate_phoneme_tts.py
import json
import time
from pathlib import Path
from gtts import gTTS
import soundfile as sf
import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "reference_audio" / "phonemes"
META_FILE = ROOT / "metadata" / "phoneme_audio_index.json"

TARGET_SR = 16000

# COMPLETE CMU Phoneme Dictionary
phoneme_examples = {
    # Vowels
    "AA0": "odd", "AA1": "father", "AA2": "odd",
    "AE0": "at", "AE1": "cat", "AE2": "at",
    "AH0": "about", "AH1": "hut", "AH2": "about",
    "AO0": "off", "AO1": "dog", "AO2": "off",
    "AW0": "out", "AW1": "cow", "AW2": "out",
    "AY0": "hide", "AY1": "my", "AY2": "hide",
    "EH0": "ed", "EH1": "red", "EH2": "ed",
    "ER0": "letter", "ER1": "bird", "ER2": "letter",
    "EY0": "okay", "EY1": "say", "EY2": "okay",
    "IH0": "it", "IH1": "sit", "IH2": "it",
    "IY0": "happy", "IY1": "see", "IY2": "happy",
    "OW0": "go", "OW1": "go", "OW2": "go",
    "OY0": "toy", "OY1": "boy", "OY2": "toy",
    "UH0": "book", "UH1": "book", "UH2": "book",
    "UW0": "you", "UW1": "too", "UW2": "you",
    
    # Consonants
    "B": "bad", "CH": "cheese", "D": "dog", "DH": "the",
    "F": "fan", "G": "go", "HH": "hat", "JH": "judge",
    "K": "cat", "L": "live", "M": "man", "N": "no",
    "NG": "sing", "P": "pen", "R": "red", "S": "sit",
    "SH": "she", "T": "top", "TH": "think", "V": "van",
    "W": "we", "Y": "yes", "Z": "zoo", "ZH": "measure"
}

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)

def generate_audio_gtts(text, out_path):
    try:
        tts = gTTS(text=text, lang='en', tld='com')
        tts.save(str(out_path))
        return True
    except Exception as e:
        print(f"  [ERROR] gTTS failed for '{text}': {e}")
        return False

def post_process_audio(in_path, target_sr=TARGET_SR):
    try:
        data, sr = librosa.load(str(in_path), sr=target_sr, mono=True)
        data, _ = librosa.effects.trim(data, top_db=25)
        maxv = np.abs(data).max()
        if maxv > 0:
            data = data / maxv * 0.95
        sf.write(str(in_path), data, sr)
        return len(data) / sr
    except Exception as e:
        print(f"  [Error] Post-processing failed for {in_path.name}: {e}")
        return 0.0

def main():
    ensure_dirs()
    metadata = {}
    print(f"Generating Complete Reference Set ({len(phoneme_examples)} phonemes)...")

    for phoneme, word in phoneme_examples.items():
        out_file = OUT_DIR / f"{phoneme}.wav"
        # Only generate if missing to save time, or force overwrite if needed
        # For now, we overwrite to ensure consistency
        success = generate_audio_gtts(word, out_file)
        
        if success and out_file.exists():
            duration = post_process_audio(out_file)
            print(f"Generated {phoneme} -> {duration:.2f}s")
            metadata[phoneme] = {
                "file": str(out_file.relative_to(ROOT)),
                "example_word": word,
                "duration_s": round(duration, 3),
                "sr": TARGET_SR,
            }
            time.sleep(0.2) 

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDONE! Complete reference audio generated.")

if __name__ == "__main__":
    main()