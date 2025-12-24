# scripts/extract_embeddings.py
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
try:
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(DEVICE)
    model.eval()
except Exception as e:
    print(f"[Extractor] Warning: Could not load model: {e}")

def extract_embedding(audio_path: str) -> np.ndarray:
    if model is None: return None
    
    try:
        # 1. Load Audio
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1: data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)

        # 2. Pad Short Audio (Crucial)
        # Wav2Vec2 conv layers reduce dimensionality. 
        # If input is too short, output is empty. We pad to ~0.1s.
        target_len = 1600 
        if len(data) < target_len:
            pad_size = target_len - len(data)
            data = np.pad(data, (0, pad_size), 'constant')

        tensor = torch.from_numpy(data).float().to(DEVICE)
        if tensor.dim() == 1: tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            # Extract features from the LAST Transformer Layer (Contextual)
            features, _ = model.extract_features(tensor)
            last_layer = features[-1] # Shape: [1, Frames, 768]
            
            # --- THE FIX: MAX POOLING ---
            # Instead of averaging (which dilutes the sound with silence),
            # we take the MAXIMUM activation across the time dimension.
            # This captures the "Peak Character" of the phoneme, ignoring misaligned edges.
            emb = last_layer.max(dim=1)[0] 
            
            emb = emb.squeeze().cpu().numpy()
            
        return emb
    except Exception as e:
        print(f"[Error] Extraction failed for {audio_path}: {e}")
        return None