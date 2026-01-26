import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa
import stable_whisper
from scipy.spatial.distance import cosine
from pathlib import Path

# --- FIX 1: Correct Path Logic ---
# Previous code pointed to scripts/models. We need the Project Root.
ROOT = Path(__file__).resolve().parent.parent 
EMB_DIR = ROOT / "models" / "embeddings"
sys.path.append(str(ROOT / "scripts"))

# Import your existing extraction logic
try:
    from extract_embeddings import extract_embedding
except ImportError:
    print("[API] Warning: extract_embeddings not found in scripts/, using fallback.")
    pass

class SpeechTherapyModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpeechTherapyModel, cls).__new__(cls)
            cls._instance._initialize_models()
        return cls._instance

    def _initialize_models(self):
        print("[Model] Loading AI Models... (This happens once)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Whisper (Alignment & Fluency)
        self.align_model = stable_whisper.load_model("medium", device=self.device)
        
        # 2. Load Wav2Vec2 (Scoring)
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.extractor = bundle.get_model().to(self.device)
        self.extractor.eval()
        
        # 3. Pre-load Reference Embeddings (The "Centroids")
        self.references = {}
        if EMB_DIR.exists():
            for f in os.listdir(EMB_DIR):
                if f.endswith(".npy"):
                    phoneme = f.replace(".npy", "")
                    self.references[phoneme] = np.load(EMB_DIR / f)
        
        count = len(self.references)
        print(f"[Model] Loaded {count} reference phonemes from {EMB_DIR}")
        if count == 0:
            print("[WARNING] No references found! Did you run build_clean_references.py?")

        # Cache for the last processed file
        self.cache = {
            "path": None,
            "phoneme_data": [],
            "word_data": [],
            "fluency": 0,
            "weak_list": []
        }

    def _extract_features(self, audio_data):
        # Max Pooling Logic
        if len(audio_data) < 1600:
            audio_data = np.pad(audio_data, (0, 1600 - len(audio_data)), 'constant')
        
        tensor = torch.from_numpy(audio_data).float().to(self.device)
        if tensor.dim() == 1: tensor = tensor.unsqueeze(0)
        
        with torch.no_grad():
            features, _ = self.extractor.extract_features(tensor)
            emb = features[-1].max(dim=1)[0]
            return emb.squeeze().cpu().numpy()

    def _process_audio(self, audio_path):
        # Cache Check
        if self.cache["path"] == audio_path:
            return

        print(f"[Model] Processing: {audio_path}")
        
        # 1. Load Audio
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            
        # 2. Alignment
        transcript = self.align_model.transcribe(audio_path, word_timestamps=True)
        
        total_duration = len(wav) / 16000
        speech_duration = 0
        phoneme_scores = []
        word_scores = []
        
        # Robust Segment Extraction
        segments = transcript.segments if hasattr(transcript, 'segments') else transcript.get('segments', [])
        
        from g2p_en import G2p
        g2p = G2p()

        for seg in segments:
            # FIX 2: Safe Word Extraction (Handles both Objects and Dicts)
            if hasattr(seg, 'words'):
                words_list = seg.words
            elif isinstance(seg, dict):
                words_list = seg.get('words', [])
            else:
                words_list = []
            
            for w in words_list:
                # Handle Object vs Dict for individual words
                if isinstance(w, dict):
                    text = w.get('word', '').strip()
                    start = float(w.get('start', 0))
                    end = float(w.get('end', 0))
                else:
                    text = getattr(w, 'word', '').strip()
                    start = float(getattr(w, 'start', 0))
                    end = float(getattr(w, 'end', 0))
                
                speech_duration += (end - start)
                
                # Get Phonemes
                raw_phonemes = g2p(text)
                clean_phonemes = [p for p in raw_phonemes if p not in [' ', 'NB', ',', '.', '!', '?']]
                
                p_len = len(clean_phonemes)
                if p_len == 0: continue
                
                slot = (end - start) / p_len
                current_word_phonemes = []
                
                for i, p_raw in enumerate(clean_phonemes):
                    p_key = ''.join([c for c in p_raw if c.isalpha()]) # Remove stress numbers
                    
                    p_start = start + (i * slot)
                    p_end = p_start + slot
                    
                    s_idx = int(p_start * 16000)
                    e_idx = int(p_end * 16000)
                    chunk = wav[s_idx:e_idx]
                    
                    score = 0
                    # Score Calculation
                    if p_key in self.references:
                        user_emb = self._extract_features(chunk)
                        ref_emb = self.references[p_key]
                        raw_sim = 1 - cosine(user_emb, ref_emb)
                        
                        # Normalize
                        score = max(0, raw_sim)
                        if score < 0.2:
                            score = score * 250
                        else:
                            norm = (score - 0.2) / 0.8
                            score = 50 + (norm * 50)
                        score = min(100, round(score, 1))
                    
                    p_data = {
                        "phoneme": p_key,
                        "score": score,
                        "word": text
                    }
                    phoneme_scores.append(p_data)
                    current_word_phonemes.append(p_data)

                if current_word_phonemes:
                    avg_w = sum([p['score'] for p in current_word_phonemes]) / len(current_word_phonemes)
                    word_scores.append({
                        "word": text,
                        "score": round(avg_w, 1),
                        "status": "Correct" if avg_w > 70 else "Mispronounced"
                    })

        # 3. Calculate Fluency
        if total_duration > 0:
            fill_ratio = speech_duration / total_duration
            fluency = (fill_ratio - 0.2) / 0.6 * 100
            fluency = max(0, min(100, fluency))
        else:
            fluency = 0

        # 4. Weak Phonemes
        weak_list = [p['phoneme'] for p in phoneme_scores if p['score'] < 60]
        weak_list = list(set(weak_list))

        self.cache["path"] = audio_path
        self.cache["phoneme_data"] = phoneme_scores
        self.cache["word_data"] = word_scores
        self.cache["fluency"] = round(fluency, 1)
        self.cache["weak_list"] = weak_list


# --- THE API FUNCTIONS ---
_backend = SpeechTherapyModel()

def calculate_phoneme_accuracy(audio_path):
    _backend._process_audio(audio_path)
    return _backend.cache["phoneme_data"]

def calculate_word_level_accuracy(audio_path):
    _backend._process_audio(audio_path)
    return _backend.cache["word_data"]

def calculate_fluency(audio_path):
    _backend._process_audio(audio_path)
    return _backend.cache["fluency"]

def weak_phonemes(audio_path):
    _backend._process_audio(audio_path)
    return _backend.cache["weak_list"]

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Update this path if needed to point to your test file
    test_file = str(ROOT / "data" / "user_audio" / "test_user2.wav")
    
    if os.path.exists(test_file):
        print("\n--- API TEST RESULTS ---")
        print(f"File: {test_file}")
        print("Fluency Score:", calculate_fluency(test_file))
        print("Weak Phonemes:", weak_phonemes(test_file))
        words = calculate_word_level_accuracy(test_file)
        if words:
            print("Word Acc sample:", words[0])
        else:
            print("No words detected.")
    else:
        print(f"Test file not found at: {test_file}")