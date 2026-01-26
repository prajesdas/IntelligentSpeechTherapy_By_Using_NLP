import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AUDIO_META_FILE = ROOT / "metadata" / "phoneme_audio_index.json"
EMB_IDX_FILE = ROOT / "metadata" / "phoneme_embeddings_index.json"
OUT_META_FILE = ROOT / "metadata" / "phoneme_metadata.json"


def compute_duration_range(base_duration: float):
    """
    Compute a min/max allowed duration around the reference.
    Currently ±40% around base duration, with a small floor.
    """
    if base_duration is None:
        return None, None

    min_d = max(0.03, base_duration * 0.6)
    max_d = base_duration * 1.4
    return round(min_d, 3), round(max_d, 3)


def main():
    with open(AUDIO_META_FILE, "r", encoding="utf-8") as f:
        audio_meta = json.load(f)

    with open(EMB_IDX_FILE, "r", encoding="utf-8") as f:
        emb_index = json.load(f)

    combined = {}

    for phoneme, info in audio_meta.items():
        base_dur = info.get("duration_s")
        min_dur, max_dur = compute_duration_range(base_dur)

        combined[phoneme] = {
            "example_word": info.get("example_word"),
            "audio_file": info.get("file"), 
            "sr": info.get("sr"),
            "base_duration_s": base_dur,
            "min_duration_s": min_dur,
            "max_duration_s": max_dur,
            "embedding_index": emb_index.get(phoneme),
        }

    OUT_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_META_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined phoneme metadata to: {OUT_META_FILE}")
    print(f"Total phonemes: {len(combined)}")


if __name__ == "__main__":
    main()
