# run_pipeline.py
import os
import sys
import subprocess
from pathlib import Path

# --- CONFIGURATION ---
# Change this path to your new audio file!
AUDIO_FILE = "data/user_audio/WhatsApp Audio 2025-12-09 at 15.14.34_0900bf20.mp3"
# ---------------------

ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable  # Uses the current python environment

def run_command(command, description):
    print(f"\n[{description}]...")
    try:
        # Run the command and capture output (streaming it to console)
        result = subprocess.run(command, check=True)
        if result.returncode == 0:
            print(f"✅ {description} Complete.")
        else:
            print(f"❌ {description} Failed!")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def main():
    print("="*50)
    print(f"🚀 STARTING SPEECH THERAPY PIPELINE")
    print(f"📂 Audio File: {AUDIO_FILE}")
    print("="*50)

    # 1. Check if audio exists
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Error: Audio file not found at {AUDIO_FILE}")
        print("Please check the path in run_pipeline.py")
        sys.exit(1)

    # --- PIPELINE STEPS ---

    # Step 1: Alignment (Audio -> Words)
    cmd_align = [
        PYTHON_EXE, "scripts/forced_alignment.py",
        "--audio", AUDIO_FILE,
        "--model", "medium"
    ]
    run_command(cmd_align, "Step 1: Forced Alignment")

    # Step 2: G2P (Words -> Phonemes)
    cmd_g2p = [PYTHON_EXE, "scripts/g2p_word_mapping.py"]
    run_command(cmd_g2p, "Step 2: Word-to-Phoneme Mapping")

    # Step 3: Timestamp Estimator (Phoneme Timing)
    cmd_ts = [PYTHON_EXE, "scripts/phoneme_timestamp_estimator.py"]
    run_command(cmd_ts, "Step 3: Phoneme Timestamp Estimation")

    # Step 4: Slicing (Audio -> Wav Slices)
    cmd_slice = [
        PYTHON_EXE, "scripts/slice_phonemes.py",
        "--audio", AUDIO_FILE
    ]
    run_command(cmd_slice, "Step 4: Slicing Audio into Phonemes")

    # Step 5: Scoring (Slices -> Vectors -> Scores)
    cmd_score_ph = [PYTHON_EXE, "scripts/score_phonemes.py"]
    run_command(cmd_score_ph, "Step 5: Scoring Phonemes")

    # Step 6: Word Aggregation (Phonemes -> Words)
    cmd_score_word = [PYTHON_EXE, "scripts/score_words.py"]
    run_command(cmd_score_word, "Step 6: Aggregating Word Scores")

    # Step 7: Final Feedback (Words -> Report)
    cmd_report = [PYTHON_EXE, "scripts/generate_feedback.py"]
    run_command(cmd_report, "Step 7: Generating Final Report")

    print("\n" + "="*50)
    print("✨ PIPELINE FINISHED SUCCESSFULLY! ✨")
    print(f"📄 Report saved to: metadata/assessment_result.json")
    print("="*50)

if __name__ == "__main__":
    main()