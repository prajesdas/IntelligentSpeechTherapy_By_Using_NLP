from flask import Blueprint, request, jsonify
from extensions import db
from .models import Sentence

sentences_bp = Blueprint("sentences", __name__)


@sentences_bp.route("/add-sentence", methods=["POST"])
def add_sentence():
    data = request.get_json()

    required_fields = ["sentence"]
    for field in required_fields:
        if field not in data:
            return jsonify({"message": f"{field} is required"}), 400

    sentence = Sentence(
        text=data["sentence"],
        source_type=data["source_type"],
        phonemes=data.get("phonemes"),
        difficulty=data.get("difficulty")
    )

    try:
        db.session.add(sentence)
        db.session.commit()
        return jsonify({
            "message": "Sentence added successfully",
            "id": sentence.id
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "message": "Failed to add sentence",
            "error": str(e)
        }), 500
        
        
@sentences_bp.route("/add-bulk-sentences", methods=["POST"])
def add_bulk_sentences():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"message": "Expected a JSON array"}), 400

    sentences = []
    for item in data:
        sentence = Sentence(
            text=item["text"],
            source_type=item["source_type"],
            phonemes=item.get("phonemes"),
            difficulty=item.get("difficulty")
        )
        sentences.append(sentence)

    try:
        db.session.bulk_save_objects(sentences)
        db.session.commit()
        return jsonify({
            "message": "Bulk upload successful",
            "count": len(sentences)
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({
            "message": "Bulk upload failed",
            "error": str(e)
        }), 500