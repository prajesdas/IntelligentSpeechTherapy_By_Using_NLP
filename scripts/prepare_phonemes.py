import json
from pathlib import Path
#import nltk
#nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from g2p_en import G2p

# paths
ROOT = Path(__file__).resolve().parent.parent   # project root
SENTENCE_FILE = ROOT / "data" / "sentences.txt"
OUTPUT_FILE = ROOT / "metadata" / "sentence_phonemes.json"

def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # remove empty lines
    return [s for s in lines if s]

def extract_phonemes(sentences):
    g2p = G2p()
    result = []

    for i, sent in enumerate(sentences, start=1):
        raw = g2p(sent)
        # g2p-en returns spaces as ' ' → filter them out
        phonemes = [p for p in raw if p.strip()]

        result.append({
            "sentence_id": i,
            "text": sent,
            "phonemes": phonemes
        })
    return result

def main():
    sentences = load_sentences(SENTENCE_FILE)
    print(f"Loaded {len(sentences)} sentences")

    data = extract_phonemes(sentences)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved phoneme data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
