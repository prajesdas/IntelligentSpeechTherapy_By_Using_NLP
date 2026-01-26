from extensions import db
from datetime import datetime

# SENTENCES TABLE

class Sentence(db.Model):
    __tablename__ = "sentences"

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    source_type = db.Column(db.String(50), nullable=False)   # assessment / practice / custom
    phonemes = db.Column(db.JSON, nullable=True)             # list of phonemes
    # weighted_phonemes = db.Column(db.JSON, nullable=True)    # phonemes with weights
    difficulty = db.Column(db.String(20), nullable=True)     # easy / medium / hard
    created_at = db.Column(db.DateTime, default=datetime.utcnow)