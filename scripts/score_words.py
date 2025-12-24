# scripts/score_words.py
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PH_SCORES = ROOT / "metadata" / "phoneme_similarity_scores.json"
TIMESTAMPS = ROOT / "metadata" / "phoneme_timestamps.json"
OUT_FILE = ROOT / "metadata" / "word_scores.json"

def main():
    print("--- Step 2: Word Scoring ---")
    
    with open(PH_SCORES, "r") as f:
        scores = json.load(f)
    with open(TIMESTAMPS, "r") as f:
        times = json.load(f)
        
    # Map scores by filename index (e.g., "0001" from "0001_cat_K.wav")
    score_map = {}
    for item in scores:
        idx = int(item["file"].split("_")[0])
        score_map[idx] = item

    word_groups = {}
    
    for idx, t in enumerate(times):
        wid = t["word_index"]
        if wid not in word_groups:
            word_groups[wid] = {"word": t["word"], "phonemes": []}
            
        # Get score data
        s_data = score_map.get(idx, {})
        accuracy = s_data.get("similarity", 0.0)
        detected = s_data.get("best_match_phoneme", t["phoneme"])
        
        # Duration calculation
        dur = float(t["end"]) - float(t["start"])
        
        word_groups[wid]["phonemes"].append({
            "phoneme": t["phoneme"],
            "score": accuracy,
            "detected": detected,
            "duration": dur
        })

    final_words = []
    for wid in sorted(word_groups.keys()):
        data = word_groups[wid]
        p_list = data["phonemes"]
        
        # Word Logic
        avg_score = np.mean([p["score"] for p in p_list])
        status = "Correct"
        if avg_score < 0.60: status = "Mispronounced"
        elif avg_score < 0.75: status = "Needs Improvement"
        
        final_words.append({
            "word": data["word"],
            "status": status,
            "score": round(avg_score * 100, 1),
            "details": p_list
        })
        
    with open(OUT_FILE, "w") as f:
        json.dump(final_words, f, indent=2)
    print(f"Saved word scores to {OUT_FILE}")

if __name__ == "__main__":
    main()