# scripts/align_phonemes_dtw.py
import json
import numpy as np
import librosa
import argparse
import sys
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
WORD_META = ROOT / "metadata" / "word_phonemes.json"
REF_DIR = ROOT / "data" / "reference_audio" / "phonemes"
OUT_FILE = ROOT / "metadata" / "phoneme_timestamps.json"

SR = 16000
HOP_LENGTH = 160  # 10ms
MIN_DURATION_SEC = 0.04  # Minimum 40ms per phoneme

def get_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

def enforce_min_duration(intervals, word_start, word_end):
    """
    Post-processing to fix 'Zero Duration' bugs.
    Ensures every segment is at least MIN_DURATION_SEC long.
    """
    total_dur = word_end - word_start
    num_ph = len(intervals)
    
    # If the word is impossibly short, fall back to linear
    if total_dur < (num_ph * MIN_DURATION_SEC):
        # Return linear spacing
        step = total_dur / num_ph
        new_ints = []
        for i in range(num_ph):
            new_ints.append((i*step, (i+1)*step))
        return new_ints

    # Convert tuples to list for modification
    starts = [i[0] for i in intervals]
    ends = [i[1] for i in intervals]
    
    # Forward Pass: Push boundaries right if segment is too small
    for i in range(num_ph):
        dur = ends[i] - starts[i]
        if dur < MIN_DURATION_SEC:
            # Need to expand. 
            diff = MIN_DURATION_SEC - dur
            ends[i] += diff
            # Push next start
            if i + 1 < num_ph:
                starts[i+1] = ends[i]
    
    # Check if we pushed past the end of the word
    # Rescale if needed to fit back into original word duration
    last_end = ends[-1]
    if last_end > (word_end - word_start):
        scale_factor = (word_end - word_start) / last_end
        starts = [s * scale_factor for s in starts]
        ends = [e * scale_factor for e in ends]

    return list(zip(starts, ends))

def align_word_dtw(user_y, phonemes):
    ref_features = []
    ph_boundaries_ref_frames = [0]
    current_frame = 0

    for ph in phonemes:
        safe_ph = "".join([c for c in ph if c.isalnum()])
        ph_path = REF_DIR / f"{safe_ph}.wav"
        
        if not ph_path.exists():
            silence_y = np.zeros(int(0.1 * SR)) 
            mfcc_silence = get_mfcc(silence_y, SR)
            ref_features.append(mfcc_silence)
            current_frame += mfcc_silence.shape[1]
            ph_boundaries_ref_frames.append(current_frame)
            continue

        y_ref, _ = librosa.load(str(ph_path), sr=SR)
        y_ref, _ = librosa.effects.trim(y_ref, top_db=20)
        mfcc_ref = get_mfcc(y_ref, SR)
        ref_features.append(mfcc_ref)
        current_frame += mfcc_ref.shape[1]
        ph_boundaries_ref_frames.append(current_frame)

    if not ref_features: return None

    full_ref_mfcc = np.concatenate(ref_features, axis=1)
    user_mfcc = get_mfcc(user_y, SR)

    try:
        D, wp = librosa.sequence.dtw(X=full_ref_mfcc, Y=user_mfcc, metric='euclidean')
    except Exception:
        return None

    wp = wp[::-1] 
    path_map = {r: u for r, u in wp}

    user_boundaries_sec = [0.0]
    for ref_boundary in ph_boundaries_ref_frames[1:]:
        closest_ref = min(path_map.keys(), key=lambda k: abs(k - ref_boundary))
        user_frame = path_map[closest_ref]
        user_boundaries_sec.append(user_frame * HOP_LENGTH / SR)

    raw_intervals = []
    for i in range(len(phonemes)):
        start = user_boundaries_sec[i] if i < len(user_boundaries_sec) else user_boundaries_sec[-1]
        end = user_boundaries_sec[i+1] if i+1 < len(user_boundaries_sec) else user_boundaries_sec[-1]
        raw_intervals.append((start, end))
            
    return raw_intervals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to user audio file")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"[ERROR] Audio not found: {audio_path}")
        return

    if not WORD_META.exists():
        print("[ERROR] word_phonemes.json missing.")
        return

    print(f"--- Starting DTW Alignment (Min Duration: {MIN_DURATION_SEC}s) ---")
    
    try:
        full_audio, _ = librosa.load(str(audio_path), sr=SR)
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        return

    with open(WORD_META, "r") as f:
        words = json.load(f)

    new_phoneme_data = []

    for entry in words:
        word = entry["word"]
        phonemes = entry["phonemes"]
        w_start = float(entry["start"])
        w_end = float(entry["end"])
        
        if not phonemes: continue

        start_idx = int(w_start * SR)
        end_idx = int(w_end * SR)
        if start_idx < 0: start_idx = 0
        if end_idx > len(full_audio): end_idx = len(full_audio)
        
        word_y = full_audio[start_idx:end_idx]
        
        # Determine strategy
        intervals = None
        # Only use DTW if we have enough audio data
        if (w_end - w_start) > 0.1 and len(word_y) > 1000:
            intervals = align_word_dtw(word_y, phonemes)

        # Apply Min Duration Logic (Whether DTW worked or not)
        if intervals:
            # Fix zero-length intervals
            intervals = enforce_min_duration(intervals, 0.0, w_end - w_start)
        else:
            # Fallback Linear
            dur = w_end - w_start
            step = dur / len(phonemes)
            intervals = [(i*step, (i+1)*step) for i in range(len(phonemes))]

        # Save
        for i, (p_start, p_end) in enumerate(intervals):
            new_phoneme_data.append({
                "word_index": entry["index"],
                "word": word,
                "phoneme": phonemes[i],
                "start": round(w_start + p_start, 4),
                "end": round(w_start + p_end, 4)
            })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(new_phoneme_data, f, indent=2)

    print(f"[SUCCESS] Alignment saved to: {OUT_FILE.name}")

if __name__ == "__main__":
    main()