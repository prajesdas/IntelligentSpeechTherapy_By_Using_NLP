# scripts/compare_phonemes.py
import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np

# Force Python to print logs immediately (Fixes silent crashes)
sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SLICES = ROOT / "data" / "user_audio_slices" / "phonemes"
DEFAULT_REF = ROOT / "models" / "embeddings"
OUT_FILE = ROOT / "metadata" / "phoneme_similarity_scores.json"

# Add the 'scripts' folder to system path so we can import 'extract_embeddings.py'
sys.path.append(str(ROOT / "scripts"))

def cosine(a, b):
    if a is None or b is None: return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def main():
    print("\n--- Starting Phoneme Comparison ---")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", default=str(DEFAULT_SLICES))
    parser.add_argument("--ref", default=str(DEFAULT_REF))
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    # Resolve paths to avoid Windows confusion
    slices_dir = Path(args.slices).resolve()
    ref_dir = Path(args.ref).resolve()
    out_path = Path(args.out).resolve()

    print(f"1. Directories:")
    print(f"   - User Slices: {slices_dir}")
    print(f"   - Reference:   {ref_dir}")
    print(f"   - Output File: {out_path}")

    # --- STEP 1: Load Extractor ---
    extractor = None
    try:
        print("\n2. Loading User Extractor...")
        # Import directly now that we added 'scripts' to path
        import extract_embeddings as mod
        extractor = mod.extract_embedding
        print(f"   [SUCCESS] Extractor function loaded.")
    except ImportError as e:
        print(f"   [WARN] Could not import extract_embeddings: {e}")
    except Exception as e:
        print(f"   [ERROR] Crash during import: {e}")

    # --- STEP 2: Load Reference Embeddings ---
    print("\n3. Loading Reference Embeddings...")
    ref_emb = {}
    if ref_dir.exists():
        for f in ref_dir.glob("*.npy"):
            try:
                ref_emb[f.stem] = np.load(f)
            except Exception as e:
                print(f"   [WARN] Failed to load {f.name}: {e}")
    
    count = len(ref_emb)
    print(f"   [INFO] Loaded {count} reference embeddings.")

    if count == 0:
        print("   [CRITICAL ERROR] No reference embeddings found in models/embeddings!")
        print("   -> Did you run 'python scripts/generate_embeddings.py'?")
        return

    # --- STEP 3: Compare ---
    print("\n4. Comparing Slices...")
    if not slices_dir.exists():
        print(f"   [ERROR] Slices directory missing: {slices_dir}")
        return

    files = list(slices_dir.glob("*.wav"))
    print(f"   [INFO] Found {len(files)} user audio slices.")

    results = []
    phoneme_scores = {}
    
    for i, wav_file in enumerate(files):
        # Print progress every 10 files so you know it's working
        if i % 10 == 0: print(f"   Processing {i}/{len(files)}...", end="\r")

        try:
            target_phoneme = wav_file.stem.split("_")[-1]
        except:
            continue

        if target_phoneme not in ref_emb:
            continue

        # Extract & Compare
        if extractor:
            user_vec = extractor(str(wav_file))
            if user_vec is not None:
                score = cosine(user_vec, ref_emb[target_phoneme])
                results.append({
                    "file": str(wav_file),
                    "phoneme": target_phoneme,
                    "similarity": round(score, 4)
                })
                phoneme_scores.setdefault(target_phoneme, []).append(score)

    print(f"   Processing {len(files)}/{len(files)} - Done!      ")

    # --- STEP 4: Save ---
    if not results:
        print("\n[WARN] No comparisons made! (Check if filenames match phonemes like 'word_PH.wav')")
    
    avg_scores = {p: round(np.mean(s), 4) for p, s in phoneme_scores.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_slice": results, "per_phoneme_avg": avg_scores}, f, indent=2)

    print(f"\n[SUCCESS] Saved results to: {out_path.name}")

if __name__ == "__main__":
    main()