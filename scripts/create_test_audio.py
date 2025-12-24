# scripts/create_test_audio.py
import torch
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

# Saves a sample user recording to data/user_audio/test_user.wav
ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "user_audio" / "test_user2.wav"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("Generating NEW sample 'User' audio...")
    device = torch.device('cpu')
    
    # Load Model
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker='v3_en',
                              trust_repo=True)
    
    # NEW TEXT: "Five big birds flew over the river. Please take this cheese to the store."
    # This tests Fricatives (F, V, S, Z) and Stops (P, B, T, D)
    text = "Five big birds flew over the river. Please take this cheese to the store."
    
    # Using speaker 'en_20' (A different random voice to simulate a new user)
    print("Synthesizing at 48kHz...")
    audio_48k = model.apply_tts(text=text, speaker='en_20', sample_rate=48000)
    
    # Convert to Numpy
    audio_np = audio_48k.squeeze().numpy()
    
    # Resample to 16000 Hz
    print("Resampling to 16kHz...")
    audio_16k = librosa.resample(audio_np, orig_sr=48000, target_sr=16000)
    
    # Save
    sf.write(str(OUT_FILE), audio_16k, 16000)
    print(f"Created new test file: {OUT_FILE}")
    print(f"Text: '{text}'")

if __name__ == "__main__":
    main()