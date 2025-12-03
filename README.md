# IntelligentSpeechTherapy_By_Using_NLP
The Intelligent Speech Therapy Platform is an AI-powered system that evaluates users’ spoken English by analyzing recorded sentences for pronunciation accuracy, fluency, and phoneme-level errors.

An AI-powered pronunciation improvement system that analyzes spoken English, detects phoneme-level errors, and provides personalized feedback and practice recommendations.

This repository contains the **NLP module** of the Intelligent Speech Therapy Platform, including phoneme extraction, alignment preparation, reference audio processing, and pronunciation scoring components.

---

## Project Overview
The Intelligent Speech Therapy Platform functions like an automated speech coach.  
It listens to user speech, analyzes phoneme-level pronunciation, identifies errors, and recommends targeted exercises.

The system aims to:
- Improve English pronunciation
- Detect mispronounced words and phonemes
- Provide adaptive practice exercises
- Track user progress over time

---

##  Core Features (NLP Side)
- **Speech-to-Text (STT)** using WhisperX  
- **Forced Alignment** to match audio → phonemes  
- **Phoneme Extraction** using g2p-en  
- **Reference Embedding Generation** using wav2vec2  
- **Pronunciation Scoring** (cosine similarity)  
- **Error Detection** for weak or incorrect phonemes  
- **Personalized Recommendations** based on user mistakes  

---

## Project Structure
IntelligentSpeechTherapy_NLP/
├── data/
│ └── sentences.txt
├── metadata/
│ └── sentence_phonemes.json
├── scripts/
│ └── prepare_phonemes.py
├── venv/
└── requirements.txt



##  Installation & Setup

### 1. Clone the project or create the folder
```bash
git clone <repo-url>
cd IntelligentSpeechTherapy_NLP
2. Create & activate virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate     # Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
