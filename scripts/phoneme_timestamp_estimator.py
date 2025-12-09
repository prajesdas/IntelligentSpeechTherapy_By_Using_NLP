# scripts/phoneme_timestamp_estimator.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORD_PH_FILE = ROOT / "metadata" / "word_phonemes.json"
OUT_FILE = ROOT / "metadata" / "phoneme_timestamps.json"

def main():
    if not WORD_PH_FILE.exists():
        print("ERROR: word_phonemes.json not found at", WORD_PH_FILE)
        return

    with open(WORD_PH_FILE, "r", encoding="utf-8") as f:
        words = json.load(f)


    phoneme_list = []
    for entry in words:
        phonemes = entry.get("phonemes", [])
        start = float(entry["start"])
        end = float(entry["end"])
        if len(phonemes) == 0:
            continue
        dur = end - start
        per = dur / len(phonemes)
        for i, p in enumerate(phonemes):
            p_start = start + i * per
            p_end = start + (i + 1) * per
            phoneme_list.append({
                "word_index": entry.get("index"),
                "word": entry.get("word"),
                "phoneme": p,
                "start": round(p_start, 6),
                "end": round(p_end, 6)
            })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(phoneme_list, f, indent=2)
    print("Saved phoneme timestamps to:", OUT_FILE)

if __name__ == "__main__":
    main()
