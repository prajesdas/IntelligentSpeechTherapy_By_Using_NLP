import librosa
import random
import logging
import traceback

# Robust import: prefer package-relative import to the Backend root; fall back to importlib
mi = None
try:
    # Go up two levels from assessments -> Backend
    from ... import model_interface as mi  # type: ignore
except Exception as e:
    logging.warning("Relative import from package failed: %s", e)
    import importlib
    try:
        mi = importlib.import_module("Backend.model_interface")
    except Exception as e2:
        logging.warning("importlib import Backend.model_interface failed: %s", e2)
        try:
            mi = importlib.import_module("model_interface")
        except Exception as e3:
            logging.exception("Failed to import model_interface by any method: %s", e3)
            mi = None

if mi is not None:
    mi_calc_phoneme_accuracy = getattr(mi, 'calculate_phoneme_accuracy', None)
    mi_calc_word_level_accuracy = getattr(mi, 'calculate_word_level_accuracy', None)
    mi_calc_fluency = getattr(mi, 'calculate_fluency', None)
    mi_weak_phonemes = getattr(mi, 'weak_phonemes', None)
else:
    def _not_available(*args, **kwargs):
        raise RuntimeError("model_interface not available: import failed")
    mi_calc_phoneme_accuracy = _not_available
    mi_calc_word_level_accuracy = _not_available
    mi_calc_fluency = _not_available
    mi_weak_phonemes = _not_available


# Wrapper functions with logging and safe defaults
def calculate_phoneme_accuracy(audio_path):
    try:
        res = mi_calc_phoneme_accuracy(audio_path)
        return res
    except Exception:
        logging.exception("calculate_phoneme_accuracy failed for %s", audio_path)
        return []

def calculate_word_level_accuracy(audio_path):
    try:
        res = mi_calc_word_level_accuracy(audio_path)
        return res
    except Exception:
        logging.exception("calculate_word_level_accuracy failed for %s", audio_path)
        return []

def calculate_fluency(audio_path):
    try:
        res = mi_calc_fluency(audio_path)
        return res
    except Exception:
        logging.exception("calculate_fluency failed for %s", audio_path)
        return 0

def weak_phonemes(audio_path):
    try:
        res = mi_weak_phonemes(audio_path)
        return res
    except Exception:
        logging.exception("weak_phonemes failed for %s", audio_path)
        return []


def run_assessment(audio_path):
    results = {}
    results['phoneme_accuracy'] = calculate_phoneme_accuracy(audio_path)
    results['word_level_accuracy'] = calculate_word_level_accuracy(audio_path)
    results['fluency'] = calculate_fluency(audio_path)
    results['weak_phonemes'] = weak_phonemes(audio_path)
    print("Assessment Results:", results)
    return results


