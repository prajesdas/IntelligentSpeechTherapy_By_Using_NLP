# scripts/prepare_phonemes.py

from pathlib import Path
import json

import nltk
from g2p_en import G2p

# Make sure NLTK resources are available
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("cmudict", quiet=True)

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "metadata" / "sentence_phonemes.json"

# You can change / extend these sentences later
SENTENCES = [
    "The red rabbit ran around the river.",
    "She sells thick shells on the shore.",
    "They think these things are easy.",
    "Please bring the blue glass bottle.",
    "The cat caught a tiny mouse.",
    "I would like a cup of coffee today.",
    "The boy bought a brown ball.",
    "We usually watch television at night.",
    "My father works in a farm far away.",
    "This zebra lives in a busy zoo.",
    "The queen quickly questioned the guard.",
    "He enjoys reading English every evening.",
]

g2p = G2p()


def extract_phonemes(sentences):
    data = []
    for i, sent in enumerate(sentences, start=1):
        raw = g2p(sent)
        # g2p_en returns mix of chars + phonemes; keep uppercase-like tokens as phonemes
        phonemes = [tok for tok in raw if tok.isalpha() and tok.upper() == tok]
        data.append(
            {
                "sentence_id": i,
                "text": sent,
                "phonemes": phonemes,
            }
        )
    return data


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(SENTENCES)} sentences")
    data = extract_phonemes(SENTENCES)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved phoneme data to", OUT_FILE)


if __name__ == "__main__":
    main()
