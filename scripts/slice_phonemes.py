# scripts/slice_phonemes.py
import soundfile as sf
import numpy as np
import json
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent.parent
PH_TS = ROOT / "metadata" / "phoneme_timestamps.json"
OUT_DIR = ROOT / "data" / "user_audio_slices" / "phonemes"

def load_audio(audio_path):
    data, sr = sf.read(str(audio_path), dtype="float32")
    # mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def write_slice(data, sr, out_path):
    # normalize and write 16-bit PCM
    maxv = np.abs(data).max()
    if maxv > 0:
        data = data / maxv * 0.98
    sf.write(str(out_path), data, sr, subtype="PCM_16")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="Path to user audio file (wav preferred)", required=False)
    args = parser.parse_args()

    if not PH_TS.exists():
        print("ERROR: phoneme_timestamps.json not found at", PH_TS)
        return

    with open(PH_TS, "r", encoding="utf-8") as f:
        phs = json.load(f)

    # audio path resolution
    if args.audio:
        audio_path = Path(args.audio)
    else:
        # try standard locations
        candidates = list(ROOT.glob("data/user_audio/*"))
        if len(candidates) == 0:
            print("ERROR: no audio provided and no files found in data/user_audio/")
            return
        # pick the newest file
        audio_path = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    if not audio_path.exists():
        print("ERROR: audio file does not exist:", audio_path)
        return

    data, sr = load_audio(audio_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, p in enumerate(phs):
        s = float(p["start"])
        e = float(p["end"])
        start_frame = int(round(s * sr))
        end_frame = int(round(e * sr))
        if start_frame < 0: start_frame = 0
        if end_frame > data.shape[0]: end_frame = data.shape[0]
        if end_frame <= start_frame:
            print(f"skip zero-length slice for {p}")
            continue
        clip = data[start_frame:end_frame]
        # filename: index_word_phoneme.wav (safe)
        safe_word = "".join(c for c in p["word"] if c.isalnum() or c in ("_","-")).lower()
        fname = f"{i:04d}_{safe_word}_{p['phoneme']}.wav"
        out_path = OUT_DIR / fname
        write_slice(clip, sr, out_path)
        saved += 1

    print(f"Saved {saved} phoneme slices to:", OUT_DIR)

if __name__ == "__main__":
    main()
