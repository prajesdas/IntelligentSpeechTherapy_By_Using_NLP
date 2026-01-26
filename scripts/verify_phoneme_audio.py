# scripts/verify_phoneme_audio.py

from pathlib import Path
import soundfile as sf
import os

ROOT = Path(__file__).resolve().parent.parent
PHONEME_DIR = ROOT / "data" / "reference_audio" / "phonemes"


def main():
    if not PHONEME_DIR.is_dir():
        print("Phoneme dir not found:", PHONEME_DIR)
        return

    files = sorted(f for f in os.listdir(PHONEME_DIR) if f.lower().endswith(".wav"))
    print(f"Found {len(files)} phoneme files.")
    for name in files:
        path = PHONEME_DIR / name
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            channels = data.shape[1]
            length = data.shape[0]
        else:
            channels = 1
            length = data.shape[0]
        dur = length / sr
        print(f"{name}: sr={sr}, duration={dur:.3f}s, channels={channels}")


if __name__ == "__main__":
    main()
