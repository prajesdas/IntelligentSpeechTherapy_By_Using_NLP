# scripts/build_clean_references.py
import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from g2p_en import G2p
import stable_whisper
import librosa
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "reference_audio" / "phonemes"
EMB_DIR = ROOT / "models" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)
# scripts/build_clean_references.py

phoneme_map = {
    # Vowels (Clear, stressed examples)
    "AA": ["bought", "saw", "call", "law", "ball", "caught"], 
    "AE": ["cat", "bat", "map", "apple", "hat", "dad"],
    "AH": ["cut", "bug", "oven", "up", "sun", "bus"],
    "AO": ["dog", "off", "lost", "cost", "long", "song"],
    "AW": ["house", "cow", "out", "loud", "mouse", "count"],
    "AY": ["like", "my", "pie", "high", "sky", "fly"],
    "EH": ["bed", "ten", "red", "head", "men", "egg"],
    "ER": ["bird", "fur", "turn", "learn", "word", "nurse"],
    "EY": ["day", "eight", "face", "play", "cake", "rain"],
    "IH": ["sit", "big", "win", "hit", "in", "it"],
    "IY": ["she", "see", "happy", "me", "tea", "key"],
    "OW": ["go", "no", "boat", "home", "road", "low"],
    "OY": ["boy", "toy", "oil", "coin", "joy", "noise"],
    "UH": ["put", "book", "look", "good", "foot", "cook"],
    "UW": ["you", "blue", "food", "moon", "new", "two"],

    # Consonants (Mix of Start, Middle, End positions)
    "B":  ["bat", "ball", "tub", "baby", "job", "rub"],
    "CH": ["church", "match", "cheese", "watch", "catch", "chair"],
    "D":  ["dog", "dad", "mad", "day", "bad", "red"],
    "DH": ["the", "this", "mother", "that", "brother", "breathe"],
    "F":  ["fan", "fish", "life", "fun", "off", "leaf"],
    "G":  ["go", "get", "big", "dog", "bag", "egg"],
    "HH": ["hat", "home", "hi", "happy", "house", "hello"],
    "JH": ["judge", "job", "age", "jar", "edge", "large"],
    "K":  ["cat", "kite", "back", "key", "sky", "rock"],
    "L":  ["let", "leg", "ball", "long", "call", "love"],
    "M":  ["man", "mom", "room", "me", "home", "mouse"],
    "N":  ["no", "nine", "sun", "now", "one", "rain"],
    "NG": ["sing", "long", "ring", "song", "king", "wing"],
    "P":  ["pen", "top", "pot", "pie", "cup", "map"],
    "R":  ["red", "run", "car", "rat", "door", "star"],
    "S":  ["sit", "sun", "bus", "see", "class", "grass"],
    "SH": ["she", "shoe", "fish", "shop", "wish", "ship"],
    "T":  ["top", "ten", "cat", "to", "hot", "sit"],
    "TH": ["think", "math", "thin", "thank", "mouth", "bath"],
    "V":  ["van", "vote", "love", "very", "five", "have"],
    "W":  ["we", "win", "water", "wet", "way", "wait"],
    "Y":  ["yes", "you", "yard", "yellow", "year", "young"],
    "Z":  ["zoo", "zebra", "nose", "zip", "rose", "size"],
    "ZH": ["measure", "vision", "beige", "usual", "pleasure", "garage"]
}

try:
    from extract_embeddings import extract_embedding
except ImportError:
    print("Error: scripts/extract_embeddings.py not found.")
    exit()

def get_silero_model():
    print("Loading Silero TTS...")
    device = torch.device('cpu')
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker='v3_en',
                              trust_repo=True)
    model.to(device)
    return model

def main():
    print("--- Building CENTROID Reference Library (Averaging Examples) ---")
    
    tts_model = get_silero_model()
    align_model = stable_whisper.load_model("medium", device="cpu")
    g2p = G2p()
    
    for target_phoneme, word_list in phoneme_map.items():
        print(f"\nProcessing {target_phoneme} (Examples: {word_list})...")
        
        collected_embeddings = []

        for word in word_list:
            try:
                # 1. Generate Audio
                audio_48k = tts_model.apply_tts(text=word, speaker='en_0', sample_rate=48000)
                audio_np = audio_48k.squeeze().numpy()
                audio_16k = librosa.resample(audio_np, orig_sr=48000, target_sr=16000)
                
                # Temp file
                tmp_wav = OUT_DIR / f"temp_{target_phoneme}_{word}.wav"
                sf.write(str(tmp_wav), audio_16k, 16000)
                
                # 2. Align
                result = align_model.transcribe(str(tmp_wav), word_timestamps=True)
                
                # Find word bounds
                word_start, word_end = 0.0, 0.0
                found = False
                
                # Robust extraction (Object or Dict)
                segments = result.segments if hasattr(result, 'segments') else result.get('segments', [])
                for seg in segments:
                    words = seg.words if hasattr(seg, 'words') else seg.get('words', [])
                    for w in words:
                        txt = getattr(w, 'word', w.get('word','')).strip().lower().replace('.','').replace(',','')
                        if txt == word.lower():
                            word_start = float(getattr(w, 'start', w.get('start', 0)))
                            word_end = float(getattr(w, 'end', w.get('end', 0)))
                            found = True
                            break
                    if found: break
                
                if not found: word_end = len(audio_16k)/16000

                # 3. Slice Logic
                p_in_word = g2p(word)
                p_in_word = [p for p in p_in_word if p not in [' ', 'NB', ',', '.', '!', '?']]
                clean_p = [''.join([c for c in p if c.isalpha()]) for p in p_in_word]
                
                if target_phoneme in clean_p:
                    p_idx = clean_p.index(target_phoneme)
                else:
                    p_idx = 0 
                
                dur = word_end - word_start
                slot = dur / len(p_in_word) if len(p_in_word) > 0 else dur
                
                start_s = int((word_start + p_idx*slot) * 16000)
                end_s = int((word_start + (p_idx+1)*slot) * 16000)
                
                # Extract & Embed
                # Buffer padding handles short vowels
                sliced = audio_16k[max(0, start_s-400):min(len(audio_16k), end_s+400)]
                slice_path = OUT_DIR / f"ref_{target_phoneme}_{word}.wav"
                sf.write(str(slice_path), sliced, 16000)
                
                emb = extract_embedding(str(slice_path))
                if emb is not None:
                    collected_embeddings.append(emb)
                
                # Cleanup
                if tmp_wav.exists(): os.remove(tmp_wav)

            except Exception as e:
                print(f"  [Skipped] {word}: {e}")
                continue

        # 4. Calculate Centroid (Average)
        if len(collected_embeddings) > 0:
            # Stack into a matrix [N, 768] and take mean across axis 0
            stack = np.vstack(collected_embeddings)
            centroid = np.mean(stack, axis=0)
            
            # Save the "Golden" Centroid
            np.save(EMB_DIR / f"{target_phoneme}.npy", centroid)
            print(f"  => Saved Centroid for {target_phoneme} (Averaged {len(collected_embeddings)} samples)")
        else:
            print(f"  [Error] No valid embeddings for {target_phoneme}")

if __name__ == "__main__":
    main()