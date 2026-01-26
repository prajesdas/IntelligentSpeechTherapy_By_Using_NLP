import os
import sys
import logging
import torch
import torchaudio
import soundfile as sf
import numpy as np
import librosa
import stable_whisper
from scipy.spatial.distance import cosine
from pathlib import Path

# --- FIX 1: Correct Path Logic ---
# Use package (Backend) directory as root to locate models and scripts reliably
ROOT = Path(__file__).resolve().parent
EMB_DIR = ROOT / "models" / "embeddings"
# Ensure scripts inside Backend are importable
sys.path.append(str(ROOT / "scripts"))

def embdir():
    return EMB_DIR

# Import your existing extraction logic (robust fallback)
extract_embedding = None
# 1) Try package-relative import (Backend.scripts.extract_embeddings)
try:
    from .scripts.extract_embeddings import extract_embedding as _extract_embedding
    extract_embedding = _extract_embedding
except Exception:
    # 2) Try absolute import from scripts package
    try:
        from scripts.extract_embeddings import extract_embedding as _extract_embedding
        extract_embedding = _extract_embedding
    except Exception:
        # 3) Try importing module names dynamically
        import importlib
        try:
            mod = importlib.import_module("scripts.extract_embeddings")
            extract_embedding = getattr(mod, "extract_embedding", None)
        except Exception:
            try:
                mod = importlib.import_module("extract_embeddings")
                extract_embedding = getattr(mod, "extract_embedding", None)
            except Exception as e:
                logging.warning("extract_embeddings import failed; extraction disabled: %s", e)
                extract_embedding = None

# 4) Final safe stub so the rest of the module can operate even if extraction is missing
if extract_embedding is None:
    def extract_embedding(audio_path: str):
        logging.warning("extract_embedding is unavailable; called with %s", audio_path)
        return None

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
        self.align_model = stable_whisper.load_model("small", device=self.device)
        
        # 2. Load Wav2Vec2 (Scoring)
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.extractor = bundle.get_model().to(self.device)
        self.extractor.eval()
        
        # 3. Pre-load Reference Embeddings (The "Centroids")
        self.references = {}
        if EMB_DIR.exists():
            for f in EMB_DIR.iterdir():
                if f.suffix == ".npy":
                    phoneme = f.stem
                    self.references[phoneme] = np.load(f)
        
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
        
        # 1. Load Audio with robust fallbacks
        wav = None
        sr = None
        try:
            wav, sr = sf.read(audio_path, dtype="float32")
        except Exception as e:
            logging.warning("soundfile failed to read %s: %s. Falling back to librosa.", audio_path, e)
            try:
                # librosa will load & resample if sr is set
                wav, sr = librosa.load(audio_path, sr=16000, mono=False)
            except Exception as e2:
                logging.warning("librosa failed to read %s: %s. Trying pydub if available.", audio_path, e2)
                try:
                    from pydub import AudioSegment
                    ad = AudioSegment.from_file(audio_path)
                    samples = np.array(ad.get_array_of_samples()).astype('float32')
                    if ad.channels > 1:
                        samples = samples.reshape((-1, ad.channels)).mean(axis=1)
                    # Normalize pydub integer samples to -1..1 depending on sample width
                    max_val = float(1 << (8 * ad.sample_width - 1))
                    wav = samples / max_val
                    sr = ad.frame_rate
                except Exception as e3:
                    logging.exception("All audio loaders failed for %s: %s", audio_path, e3)
                    # Provide a short silent buffer to avoid crashes downstream
                    wav = np.zeros(1600, dtype="float32")
                    sr = 16000

        # Ensure mono
        if wav is None:
            wav = np.zeros(1600, dtype="float32")
            sr = 16000
        if hasattr(wav, 'ndim') and wav.ndim > 1:
            wav = wav.mean(axis=1)

        # Ensure correct sample rate
        if sr != 16000:
            try:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception as e:
                logging.exception("Failed to resample %s from %s to 16000: %s", audio_path, sr, e)
                # If resample fails, reassign sr to 16000 and proceed (buffer length-based durations still work approximately)
                sr = 16000
            
        # 2. Alignment
        try:
            transcript = self.align_model.transcribe(audio_path, word_timestamps=True)
            if not transcript or not getattr(transcript, "segments", None):
                print("[Model] No segments returned by Whisper")

        except FileNotFoundError as e:
            import traceback
            print("[Model] FFmpeg/ffprobe not found on PATH. Please install FFmpeg and ensure ffmpeg and ffprobe are accessible.\nSee https://ffmpeg.org/download.html")
            print(traceback.format_exc())
            # Create an empty transcript object with segments=[] so the processing continues safely
            transcript = type("T", (), {})()
            transcript.segments = []
        except Exception:
            import traceback
            print(f"[Model] Error during alignment for {audio_path}:\n{traceback.format_exc()}")
            transcript = type("T", (), {})()
            transcript.segments = []
        
        total_duration = len(wav) / 16000
        speech_duration = 0
        phoneme_scores = []
        word_scores = []
        
        # Robust Segment Extraction
        segments = transcript.segments if hasattr(transcript, 'segments') else transcript.get('segments', [])
        
        # Initialize g2p lazily and handle missing NLTK data gracefully
        try:
            from g2p_en import G2p
            if not hasattr(self, 'g2p') or self.g2p is None:
                self.g2p = G2p()
        except Exception as e:
            logging.warning("Could not initialize G2p at import time: %s", e)
            self.g2p = None

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
                
                # Get Phonemes with robust NLTK handling
                try:
                    if getattr(self, 'g2p', None) is None:
                        from g2p_en import G2p
                        self.g2p = G2p()
                    raw_phonemes = self.g2p(text)
                except LookupError as e:
                    logging.warning("NLTK resource missing when processing word '%s': %s. Attempting automatic download.", text, e)
                    try:
                        import nltk
                        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                        nltk.download('punkt', quiet=True)
                    except Exception as nd_e:
                        logging.exception("Failed to download NLTK resources: %s", nd_e)
                    try:
                        raw_phonemes = self.g2p(text)
                    except Exception as e2:
                        logging.exception("g2p failed after downloading NLTK resources for word '%s': %s", text, e2)
                        raw_phonemes = list(text)
                except Exception as e:
                    logging.exception("g2p failed for word '%s': %s", text, e)
                    raw_phonemes = list(text)

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
    try:
        _backend._process_audio(audio_path)
        return _backend.cache["phoneme_data"]
    except Exception:
        import traceback
        print(f"[Model] Error in calculate_phoneme_accuracy for {audio_path}:\n{traceback.format_exc()}")
        return []

def calculate_word_level_accuracy(audio_path):
    try:
        _backend._process_audio(audio_path)
        return _backend.cache["word_data"]
    except Exception:
        import traceback
        print(f"[Model] Error in calculate_word_level_accuracy for {audio_path}:\n{traceback.format_exc()}")
        return []

def calculate_fluency(audio_path):
    try:
        _backend._process_audio(audio_path)
        return _backend.cache["fluency"]
    except Exception:
        import traceback
        print(f"[Model] Error in calculate_fluency for {audio_path}:\n{traceback.format_exc()}")
        return 0

def weak_phonemes(audio_path):
    try:
        _backend._process_audio(audio_path)
        return _backend.cache["weak_list"]
    except Exception:
        import traceback
        print(f"[Model] Error in weak_phonemes for {audio_path}:\n{traceback.format_exc()}")
        return []

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Update this path if needed to point to your test file
    test_file = str(ROOT / "uploads" / "assessment_audio_1766897722.wav")
    
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