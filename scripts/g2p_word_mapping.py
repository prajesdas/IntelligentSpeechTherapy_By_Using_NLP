# scripts/g2p_word_mapping.py
import json
import re
from pathlib import Path
from g2p_en import G2p

ROOT = Path(__file__).resolve().parent.parent
ALIGN_FILE = ROOT / "metadata" / "alignment.json"
OUT_FILE = ROOT / "metadata" / "word_phonemes.json"

def main():
    if not ALIGN_FILE.exists():
        print("ERROR: alignment.json not found.")
        return

    with open(ALIGN_FILE, "r", encoding="utf-8") as f:
        words = json.load(f)

    g2p = G2p()
    out = []
    
    print(f"Mapping {len(words)} words to phonemes (Removing Stress Numbers)...")

    for i, w in enumerate(words):
        raw_text = w.get("word") or w.get("text") or ""
        start = w.get("start")
        end = w.get("end")
        
        # Clean word
        clean_text = re.sub(r'[^a-zA-Z\']', '', raw_text)
        
        if not clean_text or start is None or end is None:
            continue

        raw_phonemes = g2p(clean_text)
        
        clean_phonemes = []
        for p in raw_phonemes:
            if p in [' ', 'NB', ',', '.', '!', '?']: continue
            
            # FIX: Remove digits (Stress) -> AA1 becomes AA
            # This ensures we match 'EH1' to 'EH' reference easily
            p_clean = ''.join([c for c in p if c.isalpha()]) 
            clean_phonemes.append(p_clean)

        out.append({
            "index": i,
            "word": clean_text,
            "start": start,
            "end": end,
            "phonemes": clean_phonemes
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved simplified word-to-phoneme map.")

if __name__ == "__main__":
    main()