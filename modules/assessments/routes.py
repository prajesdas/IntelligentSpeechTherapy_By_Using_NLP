from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import time
from sqlalchemy import not_
import requests
import json
import re
import numpy as np
# from config import OPENROUTER_API_KEY
from dotenv import load_dotenv
load_dotenv()
from .models import AudioRecord
# importing sentence model fomr sentence directory
# from ..sentences.models import Sentence
from modules.sentences.models import Sentence
from extensions import db
from .services import (
    calculate_phoneme_accuracy,
    calculate_word_level_accuracy,
    calculate_fluency,
    weak_phonemes
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

assess_bp = Blueprint("assessment", __name__)

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

def allowed_file(filename):
    print("Checking file:", filename)
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

@assess_bp.route("/add-sentence", methods=["POST"])
def add_sentence():
    sentence_text = request.json.get("text")
    if not sentence_text:
        return jsonify({"error": "Sentence text is required"}), 400

    new_sentence = Sentence(
        text=sentence_text,
        source_type='custom',
        difficulty='medium',
        )
    db.session.add(new_sentence)
    db.session.commit()

    return jsonify({"message": "Sentence added successfully", "sentence_id": new_sentence.id}), 201

@assess_bp.route("/upload", methods=["POST"])
def upload_audio():
    print("UPLOAD ENDPOINT HIT")

    if "audio" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["audio"]
    user_id = request.form.get("user_id")
    print(user_id)
    sentence_id = request.form.get("sentence_id")
    # print(file, "  ", user_id, "  ", sentence_id)

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not sentence_id:
        return jsonify({"error": "Sentence ID missing"}), 400

    sentence = Sentence.query.get(sentence_id)
    
    if not sentence:
        return jsonify({"error": "Invalid sentence_id"}), 404
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    if file and allowed_file(file.filename):
        _, ext = os.path.splitext(file.filename)
        
        timestamp = int(time.time()) 
        filename = secure_filename(file.filename)
        filename = f"assessment_audio_{timestamp}{ext}"
        # filename = secure_filename(file.filename)
        upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Raw outputs (may be detailed lists or summary floats)
        phoneme_data = calculate_phoneme_accuracy(file_path)
        word_data = calculate_word_level_accuracy(file_path)
        fluency_score = calculate_fluency(file_path)
        weak_phonemes_list = weak_phonemes(file_path)

        # Convert all numpy types to native Python types
        phoneme_data = convert_to_native_types(phoneme_data)
        word_data = convert_to_native_types(word_data)
        fluency_score = convert_to_native_types(fluency_score)
        weak_phonemes_list = convert_to_native_types(weak_phonemes_list)

        # Helper: derive numeric average from model outputs
        def _avg_score_from_list(data):
            try:
                if data is None:
                    return None
                if isinstance(data, list):
                    scores = []
                    for item in data:
                        if isinstance(item, dict) and 'score' in item:
                            try:
                                s = float(item.get('score', 0))
                                scores.append(s)
                            except Exception:
                                continue
                        elif isinstance(item, (int, float)):
                            scores.append(float(item))
                    if not scores:
                        return None
                    return round(sum(scores) / len(scores), 2)
                elif isinstance(data, (int, float)):
                    return float(data)
            except Exception as e:
                current_app.logger.exception("Failed to summarize scores: %s", e)
            return None

        # Convert detailed outputs to DB-friendly numeric values
        phoneme_accuracy = _avg_score_from_list(phoneme_data)
        word_accuracy = _avg_score_from_list(word_data)

        # Prepare a warning if nothing useful returned
        warning = None
        if phoneme_accuracy is None and word_accuracy is None and (fluency_score in (None, 0)) and (not weak_phonemes_list):
            current_app.logger.warning(
                "Assessment returned empty/default results for file %s", file_path
            )
            warning = "One or more assessment metrics unavailable; check server logs."

        audio_record = AudioRecord(
            user_id=user_id,
            file_path=file_path,
            sentence_id=sentence_id,
            phoneme_accuracy=phoneme_accuracy,
            word_accuracy=word_accuracy,
            fluency_score=fluency_score,
            weak_phonemes=weak_phonemes_list
        )

        # Keep raw details available in the response for debugging
        details = {
            "phoneme_details": phoneme_data,
            "word_details": word_data
        }
        db.session.add(audio_record)
        db.session.commit()

        return jsonify({
            "message": "File uploaded and assessed successfully",
            "warning": warning,
            "data": {
                "phoneme_accuracy": phoneme_accuracy,
                "word_accuracy": word_accuracy,
                "fluency_score": fluency_score,
                "weak_phonemes": weak_phonemes_list,
                "details": details
            }
        }), 201
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
    
# Sentence recomondation algorithm route
@assess_bp.route("/recommend", methods=["GET"])
def recommend_sentence():
    # try:
    #     # 1. Get user assessment history
    #     assessments = AudioRecord.query.filter_by(user_id=user_id).all()

    #     # 2. Cold start: First assessment
    #     if not assessments:
    #         sentence = Sentence.query.first()
    #         if not sentence:
    #             return jsonify({"message": "No sentences available"}), 404

    #         return jsonify({
    #             "sentence_id": sentence.id,
    #             "text": sentence.text,
    #             "difficulty": sentence.difficulty,
    #             "phonemes": sentence.phonemes,
    #             "reason": "First assessment"
    #         }), 200

    #     # 3. Sentences already completed
    #     used_sentence_ids = {a.sentence_id for a in assessments}

    #     # 4. Collect all weak phonemes
    #     weak_phonemes = set()
    #     for a in assessments:
    #         if a.weak_phonemes:
    #             weak_phonemes.update(a.weak_phonemes)

    #     # 5. Candidate sentences (not repeated)
    #     candidates = Sentence.query.filter(
    #         not_(Sentence.id.in_(used_sentence_ids))
    #     ).all()

    #     if not candidates:
    #         return jsonify({
    #             "message": "All sentences completed",
    #             "reason": "All sentence are done with this user(exhausted)"
    #         }), 200

    #     # 6. Score sentences by phoneme overlap
    #     best_sentence = None
    #     best_score = 0

    #     for sentence in candidates:
    #         if not sentence.phonemes:
    #             continue

    #         overlap = set(sentence.phonemes).intersection(weak_phonemes)
    #         score = len(overlap)

    #         if score > best_score:
    #             best_score = score
    #             best_sentence = sentence

    #     # 7. If phoneme match found
    #     if best_sentence:
    #         return jsonify({
    #             "sentence_id": best_sentence.id,
    #             "text": best_sentence.text,
    #             "difficulty": best_sentence.difficulty,
    #             "matched_phonemes": list(
    #                 set(best_sentence.phonemes).intersection(weak_phonemes)
    #             ),
    #             "phonmes": best_sentence.phonemes,
    #             "reason": "weak_phoneme_match"
    #         }), 200

    #     # 8. Fallback: no phoneme match
    #     fallback = candidates[0]
    #     return jsonify({
    #         "sentence_id": fallback.id,
    #         "text": fallback.text,
    #         "difficulty": fallback.difficulty,
    #         "phonemes": fallback.phonemes,
    #         "reason": "no_phoneme_match"
    #     }), 200

    # except Exception as e:
    #     return jsonify({
    #         "error": "Recommendation failed",
    #         "details": str(e)
    #     }), 500
       
       
    #    --------------------------------------------------------------------------
    # New recommendation algorithm based on weak phonemes input
    
    # recent_sentences = recent_sentences or []

    # avoid_block = ""
    # if recent_sentences:
    #     avoid_block = "Avoid sentences similar to:\n" + "\n".join(
    #         f"- {s}" for s in recent_sentences[-3:]  # keep minimal
    #     )

    prompt = f"""
    Generate ONE random English practice sentence (8–12 words) for speech practice.
    Avoid repeating similar sentences.
    Return ONLY valid JSON in this exact format:
    {{"text":"","difficulty":"","phonemes":[]}}

    Use ARPAbet phonemes.
    Difficulty must be easy, medium, or hard.
    """.strip()
    
    # commented
    ''' {avoid_block}
    """.strip()'''

    payload = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,          # randomness
        "max_tokens": 200
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=15
    )

    # response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    # print("Generated content:", content)

    # Attempt to extract a JSON object from the model output robustly
    def _extract_json_from_text(text):
        # Find candidate JSON object(s) using braces - try the largest first
        matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        for m in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                # try a relaxed fallback by replacing single quotes
                try:
                    return json.loads(m.replace("'", '"'))
                except Exception:
                    continue
        # Fallback: slice from first '{' to last '}' and try again
        if '{' in text and '}' in text:
            s = text[text.find('{'): text.rfind('}')+1]
            try:
                return json.loads(s)
            except Exception:
                import ast
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return None
        return None

    parsed = _extract_json_from_text(content)
    if parsed is None:
        return jsonify({"error": "Failed to parse model response", "raw": content}), 502

    # Normalize keys and types
    if isinstance(parsed, dict):
        # Accept 'phon' as alias for 'phonemes'
        if 'phon' in parsed and 'phonemes' not in parsed:
            parsed['phonemes'] = parsed.pop('phon')

        # Ensure phonemes is a list of strings
        if 'phonemes' in parsed:
            if isinstance(parsed['phonemes'], str):
                parsed['phonemes'] = [p.strip(' "[],.') for p in re.split(r'[,\s]+', parsed['phonemes']) if p.strip()]
            elif isinstance(parsed['phonemes'], (list, tuple)):
                parsed['phonemes'] = [str(p) for p in parsed['phonemes']]

        # Normalize difficulty
        diff = parsed.get('difficulty', '')
        if not isinstance(diff, str) or diff.lower() not in ('easy', 'medium', 'hard'):
            parsed['difficulty'] = 'medium'
        else:
            parsed['difficulty'] = diff.lower()

        # Validate text
        text_val = parsed.get('text', '')
        if not isinstance(text_val, str) or not text_val.strip():
            return jsonify({"error": "Model output missing or invalid 'text' field", "raw": content}), 502

        return jsonify({
            "sentence_id": None,
            "text": text_val.strip(),
            "difficulty": parsed['difficulty'],
            "phonemes": parsed.get('phonemes', []),
            "reason": "model_generated"
        }), 200

    # If model returned non-object JSON (e.g., an array), return an error
    return jsonify({"error": "Model returned non-object JSON", "raw": content}), 502

