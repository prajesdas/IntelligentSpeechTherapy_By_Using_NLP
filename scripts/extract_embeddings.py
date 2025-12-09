# scripts/extract_embeddings.py
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa

# 1. Setup Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load the Model (Same as you used for reference generation)
# We load this once so we don't reload it for every file
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(DEVICE)
model.eval()

def load_audio(path, target_sr=16000):
    """Load audio and resample to 16k to match the model."""
    data, sr = sf.read(path, dtype="float32")
    
    # Handle multi-channel (convert to mono)
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    # Resample if necessary
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        
    return data

def extract_embedding(audio_path: str) -> np.ndarray:
    """
    Reads a WAV file and returns a 1D numpy embedding.
    This function is called automatically by compare_phonemes.py
    """
    try:
        # Load audio
        waveform_np = load_audio(audio_path)
        
        # Convert to Tensor [1, T]
        tensor = torch.from_numpy(waveform_np).float().to(DEVICE)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        # Extract features
        with torch.no_grad():
            features, _ = model.extract_features(tensor)
            last_hidden_state = features[-1] # [1, Frames, 768]
            
            # Mean pooling to get a single vector [768]
            embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        return embedding
    except Exception as e:
        print(f"Error extracting embedding for {audio_path}: {e}")
        return None