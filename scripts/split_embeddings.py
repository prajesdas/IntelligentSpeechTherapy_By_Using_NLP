import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMB_FILE = ROOT / "metadata" / "phoneme_embeddings.npy"
INDEX_FILE = ROOT / "metadata" / "phoneme_embeddings_index"
OUT_DIR = ROOT / "metadata" / "embeddings"

def main():
    if not EMB_FILE.exists() or not INDEX_FILE.exists():
        print("Missing embeddings or index file.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load full matrix
    M = np.load(EMB_FILE)

    # load index: phoneme → row index
    with open(INDEX_FILE, "r") as f:
        idx = json.load(f)

    for phoneme, row in idx.items():
        vec = M[row]
        out = OUT_DIR / f"{phoneme}.npy"
        np.save(out, vec)
        print("Saved:", out)

    print("All phoneme embeddings exported!")

if __name__ == "__main__":
    main()
