# scripts/generate_embeddings.py

import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
PHONEME_DIR = ROOT / "data" / "reference_audio" / "phonemes"
META_DIR = ROOT / "metadata"
OUT_MODELS = ROOT / "models" / "embeddings"

META_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODELS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(DEVICE)
model.eval()
print("Loaded torchaudio bundle: WAV2VEC2_BASE")


def load_audio(path: Path, target_sr: int = 16000):
    data, sr = sf.read(str(path), dtype="float32")
    if sr != target_sr:
        import librosa

        data = librosa.resample(data, sr, target_sr)
        sr = target_sr
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def wav2vec_embed(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    tensor = torch.from_numpy(audio).float().to(DEVICE)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # [1, T]
    with torch.no_grad():
        features, _ = model.extract_features(tensor)
        last = features[-1]  # [1, T_frames, D]
        emb = last.mean(dim=1).squeeze().cpu().numpy()
    return emb


def main():
    files = sorted(f for f in os.listdir(PHONEME_DIR) if f.lower().endswith(".wav"))
    if not files:
        raise SystemExit(f"No wav files found in {PHONEME_DIR}")

    embeddings = []
    index_map = {}

    for i, fname in enumerate(files):
        phoneme = os.path.splitext(fname)[0]
        path = PHONEME_DIR / fname
        print(f"[{i+1}/{len(files)}] {phoneme}")
        audio, sr = load_audio(path)
        e = wav2vec_embed(audio, sr)
        embeddings.append(e)
        index_map[phoneme] = i
        np.save(OUT_MODELS / f"{phoneme}.npy", e)

    emb_matrix = np.stack(embeddings, axis=0)
    emb_path = META_DIR / "phoneme_embeddings.npy"
    idx_path = META_DIR / "phoneme_embeddings_index.json"

    np.save(emb_path, emb_matrix)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2)

    print("Saved embeddings matrix:", emb_path, "shape:", emb_matrix.shape)
    print("Saved index map:", idx_path)


if __name__ == "__main__":
    main()
