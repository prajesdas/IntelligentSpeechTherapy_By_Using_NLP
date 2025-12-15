# scripts/g2p_word_mapping.py
import json
from pathlib import Path
from g2p_en import G2p
import re

ROOT = Path(__file__).resolve().parent.parent
ALIGN_FILE = ROOT / "metadata" / "alignment.json"
OUT_FILE = ROOT / "metadata" / "word_phonemes.json"

def main():
    if not ALIGN_FILE.exists():
        print("ERROR: alignment.json not found at", ALIGN_FILE)
        return

    with open(ALIGN_FILE, "r", encoding="utf-8") as f:
        words = json.load(f)

    g2p = G2p()
    out = []
    
    # Regex to keep only valid phonemes (Arpabet is uppercase letters + optional numbers)
    # This removes '.', ',', '?', '!'
    valid_phoneme_pattern = re.compile(r"^[A-Z]+[0-9]*$")

    for i, w in enumerate(words):
        text = w.get("word") or w.get("text") or ""
        start = w.get("start")
        end = w.get("end")
        
        if text is None or start is None or end is None:
            continue

        # Get raw phonemes
        raw_phonemes = g2p(text)
        
        # Clean list: Remove spaces, pipes, and punctuation
        phonemes = [
            p for p in raw_phonemes 
            if p.strip() and valid_phoneme_pattern.match(p)
        ]
        
        out.append({
            "index": i,
            "word": text,
            "start": start,
            "end": end,
            "phonemes": phonemes
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved clean word → phonemes to:", OUT_FILE)

if __name__ == "__main__":
    main()