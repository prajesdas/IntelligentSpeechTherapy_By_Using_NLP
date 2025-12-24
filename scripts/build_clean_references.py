# scripts/build_clean_references.py
import os
import torch
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from g2p_en import G2p
import stable_whisper
import librosa
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "reference_audio" / "phonemes"
EMB_DIR = ROOT / "models" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)

# Standard word map (Stress numbers removed for simplified matching)
phoneme_map = {
    "AA": "bought", "AE": "cat", "AH": "about", "AO": "saw", "AW": "house", 
    "AY": "like", "B": "bat", "CH": "church", "D": "dog", "DH": "the", 
    "EH": "bed", "ER": "bird", "EY": "eight", "F": "fan", "G": "go", 
    "HH": "hat", "IH": "sit", "IY": "happy", "JH": "judge", "K": "cat",
    "L": "let", "M": "man", "N": "no", "NG": "sing", "OW": "go",
    "OY": "boy", "P": "pen", "R": "red", "S": "sit", "SH": "she", 
    "T": "top", "TH": "think", "UH": "put", "UW": "you", "V": "van", 
    "W": "we", "Y": "yes", "Z": "zoo", "ZH": "measure"
}

# Import extractor
try:
    from extract_embeddings import extract_embedding
except ImportError:
    print("Error: scripts/extract_embeddings.py not found.")
    exit()

def get_silero_model():
    print("Loading Silero TTS...")
    device = torch.device('cpu')
    # Force reload to ensure compatibility
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker='v3_en',
                              trust_repo=True)
    model.to(device)
    return model

def main():
    print("--- Building CLEAN Reference Library (Slicing References) ---")
    
    # 1. Load Models
    tts_model = get_silero_model()
    align_model = stable_whisper.load_model("medium", device="cpu")
    g2p = G2p()
    
    print(f"Processing {len(phoneme_map)} reference phonemes...")
    
    for target_phoneme, word in phoneme_map.items():
        print(f"Processing {target_phoneme} (in '{word}')...")
        
        # A. Generate Audio (48k -> 16k)
        try:
            audio_48k = tts_model.apply_tts(text=word, speaker='en_0', sample_rate=48000)
            audio_np = audio_48k.squeeze().numpy()
            audio_16k = librosa.resample(audio_np, orig_sr=48000, target_sr=16000)
        except Exception as e:
            print(f"[WARN] TTS failed for {word}: {e}")
            continue
        
        # Save temp file for alignment
        tmp_wav = OUT_DIR / f"temp_{target_phoneme}.wav"
        sf.write(str(tmp_wav), audio_16k, 16000)
        
        # B. Alignment
        try:
            result = align_model.transcribe(str(tmp_wav), word_timestamps=True)
        except Exception as e:
            print(f"[WARN] Alignment failed for {word}: {e}")
            continue
        
        # Find the word start/end
        word_start, word_end = 0.0, 0.0
        found_word = False
        
        # FIX: Robust segment finding handling Objects OR Dicts
        segments = result.segments if hasattr(result, 'segments') else result.get('segments', [])
        
        for seg in segments:
            # Handle 'words' attribute vs dict key
            if hasattr(seg, 'words'):
                words_list = seg.words
            elif isinstance(seg, dict):
                words_list = seg.get('words', [])
            else:
                words_list = []

            for w in words_list:
                # SAFE EXTRACTION
                if isinstance(w, dict):
                    raw_text = w.get('word', '')
                    raw_start = w.get('start')
                    raw_end = w.get('end')
                else:
                    # It's an object
                    raw_text = getattr(w, 'word', '')
                    raw_start = getattr(w, 'start', 0.0)
                    raw_end = getattr(w, 'end', 0.0)

                # Clean text
                w_text = raw_text.strip().lower().replace('.', '').replace(',', '').replace('!', '')
                
                if w_text == word.lower():
                    word_start = float(raw_start)
                    word_end = float(raw_end)
                    found_word = True
                    break
            if found_word: break
            
        if not found_word:
            # Fallback: use whole file
            word_end = len(audio_16k) / 16000
            
        # C. Estimate Phoneme Position
        phonemes_in_word = g2p(word)
        # Clean g2p output
        phonemes_in_word = [p for p in phonemes_in_word if p not in [' ', 'NB', ',', '.', '!', '?']]
        
        try:
            # Strip numbers for matching (AA1 -> AA)
            clean_target = ''.join([c for c in target_phoneme if c.isalpha()])
            clean_list = [''.join([c for c in p if c.isalpha()]) for p in phonemes_in_word]
            
            # Find index
            if clean_target in clean_list:
                p_index = clean_list.index(clean_target)
            else:
                # Fallback: take first phoneme if exact match missing
                p_index = 0
            
            # Simple Time Split
            dur = word_end - word_start
            if len(phonemes_in_word) > 0:
                p_slot = dur / len(phonemes_in_word)
            else:
                p_slot = dur
            
            p_start = word_start + (p_index * p_slot)
            p_end = p_start + p_slot
            
            # D. Slice
            start_sample = int(p_start * 16000)
            end_sample = int(p_end * 16000)
            
            # Pad a tiny bit (Buffer)
            start_sample = max(0, start_sample - 400) # -0.025s
            end_sample = min(len(audio_16k), end_sample + 400) # +0.025s
            
            phoneme_audio = audio_16k[start_sample:end_sample]
            
            # Save Sliced Reference
            ref_slice_path = OUT_DIR / f"{target_phoneme}.wav"
            sf.write(str(ref_slice_path), phoneme_audio, 16000)
            
            # E. Generate Embedding immediately
            emb = extract_embedding(str(ref_slice_path))
            if emb is not None:
                np.save(EMB_DIR / f"{target_phoneme}.npy", emb)
            
        except Exception as e:
            print(f"[WARN] Processing error for {target_phoneme}: {e}")
            continue
            
        # Cleanup
        if tmp_wav.exists(): os.remove(tmp_wav)

    print("\nDONE! Clean Reference Embeddings generated.")

if __name__ == "__main__":
    main()