#!/usr/bin/env python3
# scripts/compare_phonemes.py  (patched)
from pathlib import Path
import argparse
import json
import numpy as np
import importlib
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SLICES = ROOT / "data" / "user_audio_slices" / "phonemes"
DEFAULT_REF = ROOT / "models" / "embeddings"
OUT_FILE = ROOT / "metadata" / "phoneme_similarity_scores.json"

# ---------- cosine ----------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ---------- try to find user extractor ----------
_user_extractor = None
for mod_name in ("scripts.extract_embeddings", "extract_embeddings"):
    try:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "extract_embedding"):
            _user_extractor = mod.extract_embedding
            print(f"[INFO] Using user extractor: {mod_name}.extract_embedding")
            break
    except Exception:
        pass

# ---------- helper: load .npy next to wav ----------
def load_slice_npy_if_exists(wav_path: Path):
    npy = wav_path.with_suffix(wav_path.suffix + ".npy")  # e.g. foo.wav.npy
    if not npy.exists():
        # also try wav -> .npy with just .npy extension (foo.npy)
        alt = wav_path.with_suffix(".npy")
        if alt.exists():
            npy = alt
        else:
            return None
    try:
        return np.load(npy)
    except Exception as e:
        print(f"[WARN] Failed to load slice .npy {npy}: {e}")
        return None

# ---------- fallback wav2vec extractor (lazy import) ----------
_fallback_model = None
_fallback_proc = None
def fallback_extract(wav_path: str):
    global _fallback_model, _fallback_proc
    try:
        import torch
        import torchaudio
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
    except Exception as e:
        raise RuntimeError("Fallback embedding extraction requires torch, torchaudio, transformers") from e

    if _fallback_model is None:
        _fallback_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _fallback_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        _fallback_model.eval()
    waveform, sr = torchaudio.load(wav_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(0, keepdim=True)
    arr = waveform.squeeze(0).numpy()
    inputs = _fallback_proc(arr, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = _fallback_model(**inputs).last_hidden_state
        emb = out.mean(dim=1).squeeze(0).cpu().numpy()
    return emb

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", default=str(DEFAULT_SLICES))
    parser.add_argument("--ref", default=str(DEFAULT_REF))
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

   # Force all paths to be absolute to avoid relative_to() errors
    slices_dir = Path(args.slices).resolve()
    ref_dir = Path(args.ref).resolve()
    out_path = Path(args.out).resolve()

    if not slices_dir.exists():
        print("[ERROR] slices directory not found:", slices_dir)
        sys.exit(1)
    if not ref_dir.exists():
        print("[ERROR] reference embeddings dir not found:", ref_dir)
        sys.exit(1)

    # load all .npy reference embeddings keyed by stem (AA1, B, etc)
    ref = {}
    for f in sorted(ref_dir.glob("*.npy")):
        try:
            ref[f.stem] = np.load(f)
        except Exception as e:
            print(f"[WARN] failed to load ref {f}: {e}")
    if not ref:
        print("[ERROR] no reference .npy embeddings found in", ref_dir)
        sys.exit(1)

    all_wavs = sorted(slices_dir.glob("*.wav"))
    if not all_wavs:
        print("[ERROR] no phoneme WAV slices found in", slices_dir)
        sys.exit(1)

    per_slice_results = []
    per_phoneme_scores = {}
    skipped_no_ref = 0
    skipped_no_emb = 0
    used_user_extractor = False
    used_fallback = False

    for wav in all_wavs:
        # parse phoneme from filename, assume last underscore part before extension
        name = wav.name
        try:
            phon = name.rsplit("_", 1)[-1].replace(".wav", "")
        except Exception:
            phon = None

        emb = None

        # 1) slice .npy precomputed?
        emb = load_slice_npy_if_exists(wav)
        if emb is not None:
            # already loaded
            pass
        else:
            # 2) user extractor?
            if _user_extractor is not None:
                try:
                    emb = _user_extractor(str(wav))
                    used_user_extractor = True
                    if emb is None:
                        print(f"[WARN] user extractor returned None for {wav}")
                        emb = None
                except Exception as e:
                    print(f"[WARN] user extractor raised for {wav}: {e}")
                    emb = None

        # 3) fallback model if still None
        if emb is None:
            try:
                emb = fallback_extract(str(wav))
                used_fallback = True
            except Exception:
                # fallback missing -> will skip this slice (but continue)
                emb = None

        if emb is None:
            skipped_no_emb += 1
            print(f"[SKIP] Could not obtain embedding for {wav} (no .npy, no extractor, no fallback).")
            continue

        # ensure phoneme exists in ref
        if not phon or phon not in ref:
            skipped_no_ref += 1
            print(f"[SKIP] no reference embedding for phoneme '{phon}' (file: {wav.name})")
            continue

        sim = cosine(np.asarray(emb), ref[phon])
        per_slice_results.append({
            "file": str(wav.relative_to(ROOT)),
            "phoneme": phon,
            "similarity": round(sim, 4)
        })
        per_phoneme_scores.setdefault(phon, []).append(sim)

    per_phoneme_avg = {p: round(float(np.mean(vals)), 4) for p, vals in per_phoneme_scores.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_slice": per_slice_results, "per_phoneme_avg": per_phoneme_avg}, f, indent=2)

    print("Saved similarity results to:", out_path)
    print("Slices compared:", len(per_slice_results))
    print("Skipped (no embedding):", skipped_no_emb)
    print("Skipped (no ref embedding):", skipped_no_ref)
    print("Used user extractor:", bool(used_user_extractor))
    print("Used fallback wav2vec2:", bool(used_fallback))


if __name__ == "__main__":
    main()