from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import time
import numpy as np

from extensions import db
from modules.sentences.models import Sentence
from modules.assessments.services import (
    calculate_phoneme_accuracy,
    calculate_word_level_accuracy,
    calculate_fluency
)

def convert_to_native_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    return obj

routes_bp = Blueprint('practice', __name__)
# practice sentence recommendation for the chosen weak phonemes
@routes_bp.route('/recommendation', methods=['GET', 'POST'])
def recommend_practice_sentences():
    # Support GET with query param `phonemes=AE,IH` or POST JSON {"weak_phonemes": [...]}.
    if request.method == 'GET':
        phonemes_q = request.args.get('phonemes', '')
        weak_phonemes = [p.strip() for p in phonemes_q.split(',') if p.strip()]
    else:
        data = request.get_json() or {}
        weak_phonemes = data.get('weak_phonemes', [])

    if not weak_phonemes:
        return jsonify({"message": "weak_phonemes is required"}), 400

    weak_set = set(weak_phonemes)

    # Score sentences by overlap with provided weak phonemes
    candidates = Sentence.query.filter(Sentence.phonemes != None).all()
    scored = []
    for s in candidates:
        if not s.phonemes:
            continue
        overlap = set(s.phonemes).intersection(weak_set)
        score = len(overlap)
        if score > 0:
            scored.append((score, s))

    # Sort by score desc and return top 10
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:1]]

    recommendations = [
        {
            "sentence_id": s.id,
            "text": s.text,
            "difficulty": s.difficulty,
            "phonemes": s.phonemes,
            "matched": list(set(s.phonemes).intersection(weak_set))
        }
        for s in top
    ]

    return jsonify({"recommendations": recommendations}), 200


# Submit practice audio and return assessment using existing services (no DB persistence by default)
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@routes_bp.route('/submit', methods=['POST'])
def submit_practice_audio():
    # Expects multipart/form-data with 'audio' file and 'sentence_id' (and optional 'user_id')
    if 'audio' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['audio']
    sentence_id = request.form.get('sentence_id')
    # user_id = request.form.get('user_id')  # Not used for practice submissions

    if not sentence_id:
        return jsonify({"error": "sentence_id is required"}), 400

    sentence = Sentence.query.get(sentence_id)
    if not sentence:
        return jsonify({"error": "Invalid sentence_id"}), 404

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        _, ext = os.path.splitext(file.filename)
        timestamp = int(time.time())
        filename = secure_filename(file.filename)
        filename = f"practice_audio_{timestamp}{ext}"
        upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Call assessment services
        phoneme_accuracy = calculate_phoneme_accuracy(file_path)
        word_accuracy = calculate_word_level_accuracy(file_path)
        fluency_score = calculate_fluency(file_path)

        # Convert all numpy types to native Python types
        phoneme_accuracy = convert_to_native_types(phoneme_accuracy)
        word_accuracy = convert_to_native_types(word_accuracy)
        fluency_score = convert_to_native_types(fluency_score)

        # Return results; do not persist by default (keeps practice separate)
        matched_phonemes = []
        try:
            computed_weak = []
            # Some services might provide weak phonemes; call weak_phonemes if available
            from modules.assessments.services import weak_phonemes as get_weak
            computed_weak = get_weak(file_path)
            computed_weak = convert_to_native_types(computed_weak)
        except Exception:
            computed_weak = []

        if sentence.phonemes:
            matched_phonemes = list(set(sentence.phonemes).intersection(set(computed_weak)))

        return jsonify({
            "message": "Practice submission evaluated",
            "data": {
                "phoneme_accuracy": phoneme_accuracy,
                "word_accuracy": word_accuracy,
                "fluency_score": fluency_score,
                "computed_weak_phonemes": computed_weak,
                # "matched_sentence_phonemes": matched_phonemes,
                "sentence_id": sentence.id,
                # "sentence_text": sentence.text
            }
        }), 201

    return jsonify({"error": "Invalid file type"}), 400
    