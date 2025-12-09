<<<<<<< HEAD
import os
os.environ["XDG_CACHE_HOME"] = "H:/whisper_cache"
os.environ["TRANSFORMERS_CACHE"] = "H:/whisper_cache"

import stable_whisper
import json

audio_path = r"H:\IntelligentSpeechTherapy_NLP\data\user_audio\audio.opus"

model = stable_whisper.load_model("medium", device="cpu")

result = model.transcribe(
    audio_path,
    word_timestamps=True,
    vad=True,
    language="en"
)

words = []

for segment in result.segments:
    if hasattr(segment, "words") and segment.words:
        for w in segment.words:
            words.append({
                "word": w.word,      # <-- FIX
                "start": w.start,    # <-- FIX
                "end": w.end         # <-- FIX
            })

with open("metadata/alignment.json", "w", encoding="utf-8") as f:
    json.dump(words, f, indent=2)

print("Alignment saved to metadata/alignment.json")
=======
# scripts/forced_alignment.py
import argparse
import json
import os
import sys
from pathlib import Path

# CLI
parser = argparse.ArgumentParser(description="Transcribe audio & extract word timestamps (stable_whisper)")
parser.add_argument("--audio", required=True, help="Path to input audio (mp4, mp3, wav, opus...)")
parser.add_argument("--model", default="medium", help="model size: small / base / medium / large")
parser.add_argument("--out", default="metadata/alignment.json", help="output json path (word timestamps)")
parser.add_argument("--device", default="cpu", help="pytorch device (cpu or cuda)")
parser.add_argument("--cache", default=None, help="optional cache directory (XDG_CACHE_HOME & TRANSFORMERS_CACHE)")
parser.add_argument("--language", default="en", help="language code (en)")
parser.add_argument("--save-full", action="store_true", help="also save full transcription JSON to <out>.full.json")
args = parser.parse_args()

audio_path = Path(args.audio).expanduser().resolve()
out_file = Path(args.out).expanduser().resolve()
model_name = args.model
device = args.device
language = args.language

# set caches (optional) — stable_whisper / transformers read these env vars
if args.cache:
    cache_dir = Path(args.cache).expanduser().resolve()
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
else:
    cache_dir = None

# deferred imports with helpful errors
try:
    import stable_whisper
except Exception as e:
    print("Error: stable_whisper import failed.", file=sys.stderr)
    print("Install: pip install stable-ts  AND  pip install git+https://github.com/m-bain/stable-whisper.git", file=sys.stderr)
    print("Exception:", e, file=sys.stderr)
    sys.exit(2)

# basic checks
if not audio_path.exists():
    print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
    sys.exit(3)

out_file.parent.mkdir(parents=True, exist_ok=True)

# load model
print(f"Loading stable_whisper model '{model_name}' on device={device} (cache={cache_dir}) ...")
try:
    # NOTE: do NOT pass cache_dir to load_model (some stable_whisper versions don't accept that kwarg).
    model = stable_whisper.load_model(model_name, device=device)
except Exception as e:
    print("Failed to load model. Possible causes: no disk space, corrupted cache, incompatible torch/numpy.", file=sys.stderr)
    print("Exception:", e, file=sys.stderr)
    sys.exit(4)

# transcribe / align
print("Transcribing & aligning... (this may take some time)")
try:
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad=True,
        language=language
    )
except Exception as e:
    print("Transcription failed:", e, file=sys.stderr)
    sys.exit(5)

# Optionally save full result for debugging
if args.save_full:
    try:
        full_out = out_file.with_name(out_file.stem + ".full.json")
        if hasattr(result, "to_dict"):
            serial = result.to_dict()
        elif isinstance(result, dict):
            serial = result
        else:
            try:
                serial = json.loads(json.dumps(result, default=lambda o: o.__dict__))
            except Exception:
                serial = {"note": "could not fully serialize result"}
        with open(full_out, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2)
        print(f"Saved full transcription to: {full_out}")
    except Exception as e:
        print("Warning: failed to save full result:", e, file=sys.stderr)

# extract words robustly (supports multiple result shapes)
words_out = []
segments = getattr(result, "segments", None)
if segments is None:
    if isinstance(result, dict) and "segments" in result:
        segments = result["segments"]
    else:
        print("No segments found in result; aborting.", file=sys.stderr)
        sys.exit(6)

for seg in segments:
    ws = None
    if hasattr(seg, "words"):
        ws = seg.words
    elif isinstance(seg, dict) and seg.get("words") is not None:
        ws = seg.get("words")
    else:
        ws = []

    for w in (ws or []):
        try:
            if isinstance(w, dict):
                text = w.get("text") or w.get("word") or w.get("word_text") or ""
                start = w.get("start")
                end = w.get("end")
            else:
                text = getattr(w, "word", getattr(w, "text", ""))
                start = getattr(w, "start", None)
                end = getattr(w, "end", None)

            if not text:
                continue
            text_clean = str(text).strip()
            if text_clean == "":
                continue
            if start is None or end is None:
                continue

            words_out.append({
                "word": text_clean,
                "start": float(start),
                "end": float(end)
            })
        except Exception:
            continue

# save words
try:
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(words_out, f, indent=2)
    print(f"Saved {len(words_out)} word timestamps to: {out_file}")
except Exception as e:
    print("Failed to save output file:", e, file=sys.stderr)
    sys.exit(7)

sys.exit(0)
>>>>>>> 654c855 (day-6)
