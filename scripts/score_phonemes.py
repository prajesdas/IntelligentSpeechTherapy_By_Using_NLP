# scripts/score_phonemes.py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import your extractor
try:
    from extract_embeddings import extract_embedding
except ImportError:
    print("Error: scripts/extract_embeddings.py not found.")
    exit()

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "models" / "embeddings"
USER_SLICES_DIR = ROOT / "data" / "user_audio_slices" / "phonemes"
OUT_FILE = ROOT / "metadata" / "phoneme_similarity_scores.json"

def cosine_similarity(a, b):
    if a is None or b is None: return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def main():
    print("--- Step 1: Phoneme Scoring ---")
    
    # Load Reference Embeddings
    refs = {}
    for f in EMBEDDINGS_DIR.glob("*.npy"):
        refs[f.stem] = np.load(f)
    
    results = []
    
    files = sorted(list(USER_SLICES_DIR.glob("*.wav")))
    print(f"Scoring {len(files)} slices...")

    for wav_file in tqdm(files):
        # Filename format: 0001_word_PH.wav
        parts = wav_file.stem.split("_")
        target_ph = parts[-1] 
        
        # 1. Extract User Vector
        user_vec = extract_embedding(str(wav_file))
        
        # 2. Compare with Target
        score = 0.0
        if target_ph in refs:
            score = cosine_similarity(user_vec, refs[target_ph])
            
        # 3. Detect Substitution (Compare against ALL refs)
        best_match = target_ph
        best_match_score = score
        
        for r_ph, r_vec in refs.items():
            sim = cosine_similarity(user_vec, r_vec)
            if sim > best_match_score:
                best_match_score = sim
                best_match = r_ph
        
        results.append({
            "file": str(wav_file.name),
            "phoneme": target_ph,
            "similarity": round(score, 4),
            "best_match_phoneme": best_match,
            "best_match_score": round(best_match_score, 4)
        })

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved phoneme scores to {OUT_FILE}")

if __name__ == "__main__":
    main()