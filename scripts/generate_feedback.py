# scripts/generate_feedback.py
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORD_SCORES = ROOT / "metadata" / "word_scores.json"
OUT_FILE = ROOT / "metadata" / "assessment_result.json"

def normalize_score(raw_similarity):
    """
    Maps raw cosine similarity (-1.0 to 1.0) to a human percentage (0-100).
    In Wav2Vec2 space:
      0.3 is usually 'Okay'
      0.5 is 'Good'
      0.7+ is 'Excellent'
    """
    # 1. Clip negative scores to 0
    score = max(0, raw_similarity)
    
    # 2. Apply non-linear scaling (Sigmoid-like curve)
    # This pushes 0.4 up to ~75% and 0.6 up to ~90%
    # Formula: 100 * (score ^ 0.5) is a simple way to boost mid-range scores
    # Let's use a slightly steeper curve for better distinction
    
    if score < 0.2:
        final = score * 250  # 0.1 -> 25% (Bad)
    else:
        # Map 0.2...1.0 to 50...100
        # (score - 0.2) / 0.8  --> gives 0 to 1 range
        norm = (score - 0.2) / 0.8
        final = 50 + (norm * 50)
        
    return min(100, round(final, 1))

def main():
    print("--- Step 7: Generating Feedback Report (with Normalization) ---")
    
    if not WORD_SCORES.exists():
        print("Error: word_scores.json not found.")
        return

    with open(WORD_SCORES, "r") as f:
        words = json.load(f)
        
    overall_sum = 0
    mistakes = []
    weak_phonemes = []
    weak_words = []
    
    processed_words = []
    
    for w in words:
        # Re-calculate word score based on normalized phoneme scores
        ph_scores = []
        for p in w["details"]:
            raw = p["score"]
            norm_score = normalize_score(raw)
            p["score"] = norm_score # Update the detail with human score
            ph_scores.append(norm_score)
            
            # Logic for mistakes
            if norm_score < 60.0:
                weak_phonemes.append(p["phoneme"])
                issue = "Unclear"
                if p["phoneme"] != p["detected"]:
                    issue = f"Sounded like {p['detected']}"
                mistakes.append(f"In '{w['word']}', {p['phoneme']} ({int(norm_score)}%) - {issue}")

        # Average the NORMALIZED scores for the word
        word_avg = sum(ph_scores) / len(ph_scores) if ph_scores else 0
        w["score"] = round(word_avg, 1)
        w["status"] = "Correct" if word_avg >= 70 else "Mispronounced"
        
        if w["status"] != "Correct":
            weak_words.append(w["word"])
            
        overall_sum += word_avg
        processed_words.append(w)

    final_score = round(overall_sum / len(words), 1) if words else 0
    
    # Remove duplicates
    weak_phonemes = list(set(weak_phonemes))
    
    report = {
        "Overall_score": final_score,
        "Week_phonemes": weak_phonemes,
        "Week_words": weak_words,
        "Analysis_Summary": mistakes[:5], 
        "Detailed_Word_Report": processed_words
    }
    
    with open(OUT_FILE, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*30)
    print(f"FINAL HUMAN SCORE: {final_score}%")
    print("="*30)
    print(f"Report saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()