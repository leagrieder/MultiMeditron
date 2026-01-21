"""
Wikipedia Multilingual Dataset Splitter

Splits a multilingual Wikipedia JSONL dataset into separate language-specific files
formatted for Meditron pretraining. Each output file contains all Wikipedia articles
for a single language in the Meditron pretraining format.

Input: Multilingual Wikipedia JSONL with 'text' and 'lang' fields
Output: Separate files per language: wikipedia_<lang>_pretraining.jsonl

Each output line follows the format:
    {"text": "<natural language text>", "modalities": []}

Usage:
    python convert_cleanWiki_to_pretraining.py
"""

import json
from pathlib import Path


IN_PATH = Path("../../../nemo/datasets/polyglot/clean_wikipedia/train.jsonl")
OUT_DIR = Path("src/multimeditron/translation/datasets/formatted_datasets/general_datasets/wikipedia")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ENCODING = "utf-8"


def split_by_language(in_path: Path, out_dir: Path):
    """
    Split multilingual Wikipedia dataset into separate files by language.
    Converts to Meditron pretraining format with text and empty modalities.
    """
    total_in, total_out = 0, 0
    writers = {}

    try:
        with open(in_path, "r", encoding=ENCODING, errors="ignore") as fin:
            for line in fin:
                total_in += 1
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = str(ex.get("text", "")).strip()
                lang = ex.get("lang", "").strip()
                if not text or not lang:
                    continue

                if lang not in writers:
                    out_path = out_dir / f"wikipedia_{lang}_pretraining.jsonl"
                    writers[lang] = open(out_path, "w", encoding=ENCODING)
                    print(f"🆕 Created {out_path}")

                json.dump({"text": text, "modalities": []},
                          writers[lang], ensure_ascii=False)
                writers[lang].write("\n")
                total_out += 1

                if total_in % 100000 == 0:
                    print(f"📊 Processed {total_in:,} lines... ({len(writers)} langs active)")

    finally:
        for w in writers.values():
            w.close()

    print("=" * 80)
    print(f"✅ Completed: {total_out:,}/{total_in:,} valid samples")
    print(f"🌍 Languages generated: {', '.join(sorted(writers.keys()))}")
    print(f"📁 Output folder: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    if not IN_PATH.exists():
        raise FileNotFoundError(f"❌ Input file not found: {IN_PATH}")
    split_by_language(IN_PATH, OUT_DIR)