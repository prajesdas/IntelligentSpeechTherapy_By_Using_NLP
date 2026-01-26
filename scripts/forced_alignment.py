# scripts/forced_alignment.py
import argparse
import json
import os
import sys
from pathlib import Path
import stable_whisper
import torch

# CLI Arguments
parser = argparse.ArgumentParser(description="Transcribe audio & extract word timestamps")
parser.add_argument("--audio", required=True, help="Path to input audio")
parser.add_argument("--model", default="medium", help="Whisper model size")
parser.add_argument("--out", default="metadata/alignment.json", help="Output JSON path")
parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
args = parser.parse_args()

audio_path = Path(args.audio).resolve()
out_file = Path(args.out).resolve()

def main():
    print(f"--- Step 1: Forced Alignment (Model: {args.model}) ---")
    
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        sys.exit(1)

    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    try:
        model = stable_whisper.load_model(args.model, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 2. Transcribe
    print(f"Aligning audio: {audio_path.name}...")
    try:
        result = model.transcribe(str(audio_path), word_timestamps=True)
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        sys.exit(1)

    # 3. Extract Words
    words_out = []
    
    # robust extraction handling different result formats
    segments = result.segments if hasattr(result, 'segments') else result.get('segments', [])
    
    for seg in segments:
        words = seg.words if hasattr(seg, 'words') else seg.get('words', [])
        for w in words:
            # Handle object vs dict access
            text = getattr(w, 'word', None) or w.get('word')
            start = getattr(w, 'start', None) or w.get('start')
            end = getattr(w, 'end', None) or w.get('end')
            
            if text:
                words_out.append({
                    "word": text.strip(),
                    "start": float(start),
                    "end": float(end)
                })

    # 4. Save
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(words_out, f, indent=2)

    print(f"Saved {len(words_out)} aligned words to {out_file}")

if __name__ == "__main__":
    main()