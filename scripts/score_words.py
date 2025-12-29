# scripts/score_words.py
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PH_SCORES = ROOT / "metadata" / "phoneme_similarity_scores.json"
TIMESTAMPS = ROOT / "metadata" / "phoneme_timestamps.json"
OUT_FILE = ROOT / "metadata" / "word_scores.json"

# WEIGHTS
# Vowels and Start/End consonants are critical for intelligibility
WEIGHTS = {
    # Strong Vowels = 1.0 (Standard)
    # Plosives (Critical for clarity)
    'P': 1.2, 'B': 1.2, 'T': 1.2, 'D': 1.2, 'K': 1.2, 'G': 1.2,
    # Fricatives (Often mispronounced)
    'S': 1.1, 'Z': 1.1, 'F': 1.1, 'V': 1.1, 'TH': 1.2, 'DH': 1.1,
    # Liquides/Glides
    'R': 1.1, 'L': 1.0, 'W': 1.0, 'Y': 1.0,
    # Schwas/Weak Vowels (Less penalty if wrong)
    'AH0': 0.6, 'IH0': 0.7, 'UH0': 0.7
}

def get_weight(phoneme):
    # Check exact match first (e.g. AH0)
    if phoneme in WEIGHTS: return WEIGHTS[phoneme]
    # Check base (e.g. P)
    base = ''.join([c for c in phoneme if c.isalpha()])
    return WEIGHTS.get(base, 1.0) # Default weight 1.0

def main():
    print("--- Step 6: Weighted Word Scoring ---")
    
    with open(PH_SCORES, "r") as f:
        scores = json.load(f)
    with open(TIMESTAMPS, "r") as f:
        times = json.load(f)
        
    score_map = {int(item["file"].split("_")[0]): item for item in scores}
    word_groups = {}
    
    for idx, t in enumerate(times):
        wid = t["word_index"]
        if wid not in word_groups:
            word_groups[wid] = {"word": t["word"], "phonemes": []}
            
        s_data = score_map.get(idx, {})
        accuracy = s_data.get("similarity", 0.0)
        
        word_groups[wid]["phonemes"].append({
            "phoneme": t["phoneme"],
            "score": accuracy,
            "weight": get_weight(t["phoneme"]), # Add weight
            "detected": s_data.get("best_match_phoneme", t["phoneme"])
        })

    final_words = []
    for wid in sorted(word_groups.keys()):
        data = word_groups[wid]
        p_list = data["phonemes"]
        
        # WEIGHTED AVERAGE FORMULA
        total_score = 0
        total_weight = 0
        
        for p in p_list:
            total_score += (p["score"] * p["weight"])
            total_weight += p["weight"]
            
        avg_score = total_score / total_weight if total_weight > 0 else 0
        
        status = "Correct"
        if avg_score < 0.60: status = "Mispronounced"
        
        # Pass raw weighted score (Generate Feedback script handles normalization)
        final_words.append({
            "word": data["word"],
            "status": status,
            "score": avg_score, 
            "details": p_list
        })
        
    with open(OUT_FILE, "w") as f:
        json.dump(final_words, f, indent=2)
    print(f"Saved weighted word scores to {OUT_FILE}")

if __name__ == "__main__":
    main()