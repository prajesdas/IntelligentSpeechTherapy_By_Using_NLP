# scripts/phoneme_timestamp_estimator.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORD_PH_FILE = ROOT / "metadata" / "word_phonemes.json"
OUT_FILE = ROOT / "metadata" / "phoneme_timestamps.json"

# Vowels usually take longer to say than consonants
VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
}

def get_weight(phoneme):
    # Strip numbers (AE1 -> AE)
    p = ''.join([c for c in phoneme if c.isalpha()])
    if p in VOWELS:
        return 1.5  # Vowels get 50% more time
    return 0.8      # Consonants get less time

def main():
    if not WORD_PH_FILE.exists():
        print("ERROR: word_phonemes.json not found.")
        return

    with open(WORD_PH_FILE, "r", encoding="utf-8") as f:
        words = json.load(f)

    phoneme_list = []
    
    print("Estimating timestamps using Weighted Durations (Vowels > Consonants)...")

    for entry in words:
        phonemes = entry.get("phonemes", [])
        start = float(entry["start"])
        end = float(entry["end"])
        
        if not phonemes: continue
        
        word_duration = end - start
        
        # 1. Calculate Total Weight
        weights = [get_weight(p) for p in phonemes]
        total_weight = sum(weights)
        
        if total_weight == 0: total_weight = 1 # Safety
        
        # 2. Calculate Time per Unit
        time_unit = word_duration / total_weight
        
        # 3. Assign Times
        current_time = start
        for i, p in enumerate(phonemes):
            # Duration for this specific phoneme
            p_dur = weights[i] * time_unit
            
            p_start = current_time
            p_end = current_time + p_dur
            
            # Update tracker
            current_time = p_end
            
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
    print("Saved smart timestamps to:", OUT_FILE)

if __name__ == "__main__":
    main()