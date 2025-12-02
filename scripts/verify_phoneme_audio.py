from pathlib import Path
import os
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "reference_audio" / "phonemes"

def verify():
    files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
    print("Found", len(files), "phoneme files.")
    for f in files:
        p = AUDIO_DIR / f
        data, sr = sf.read(p, dtype='float32')
        dur = data.shape[0] / sr
        channels = 1 if data.ndim == 1 else data.shape[1]
        print(f"{f}: sr={sr}, duration={dur:.3f}s, channels={channels}")

if __name__ == "__main__":
    verify()
