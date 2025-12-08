import os
os.environ["XDG_CACHE_HOME"] = "H:/whisper_cache"
os.environ["TRANSFORMERS_CACHE"] = "H:/whisper_cache"

import stable_whisper
import json

audio_path = r"H:\IntelligentSpeechTherapy_NLP\data\user_audio\audio.opus"

model = stable_whisper.load_model("medium", device="cpu")

result = model.transcribe(
    audio_path,
    word_timestamps=True,
    vad=True,
    language="en"
)

words = []

for segment in result.segments:
    if hasattr(segment, "words") and segment.words:
        for w in segment.words:
            words.append({
                "word": w.word,      # <-- FIX
                "start": w.start,    # <-- FIX
                "end": w.end         # <-- FIX
            })

with open("metadata/alignment.json", "w", encoding="utf-8") as f:
    json.dump(words, f, indent=2)

print("Alignment saved to metadata/alignment.json")
