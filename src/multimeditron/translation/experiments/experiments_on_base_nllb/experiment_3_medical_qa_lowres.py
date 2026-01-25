#!/usr/bin/env python3
"""
African Languages Experiment Using Pre-Translated Dataset

Compares translation pipeline vs native multilingual for African languages:
1. Translation pipeline (NLLB → English → Meditron)
2. Native multilingual (Direct multilingual Meditron)

Uses pre-translated medical MCQ dataset in African languages (Amharic, Hausa, Swahili, Yoruba, Zulu).

Default: 1000 samples per language

Output: src/multimeditron/translation/experiments/results/base_nllb/experiment_3_results.json
"""

import os
import sys
import re
import json
import argparse
import gc
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multimeditron.model.model import MultiModalModelForCausalLM, ChatTemplate
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.translation.translator import NLLBTranslator

AFRICAN_LANGUAGES = {
    'am': ('amh_Ethi', 'Amharic'),
    'ha': ('hau_Latn', 'Hausa'),
    'sw': ('swh_Latn', 'Swahili'),
    'yo': ('yor_Latn', 'Yoruba'),
    'zu': ('zul_Latn', 'Zulu'),
}

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"


class SimpleDataset:
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


def load_translated_dataset(json_path: str) -> SimpleDataset:
    print(f"Loading translated dataset from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} translated MCQs")
    return SimpleDataset(data)


def explore_dataset(dataset: SimpleDataset, num_samples: int = 5) -> bool:
    print("\n" + "=" * 70)
    print("DATASET EXPLORATION")
    print("=" * 70)

    print(f"\nTotal samples: {len(dataset)}")

    languages = Counter(sample.get('language') for sample in dataset)
    print("\nLanguage distribution:")
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1]
        print(f"  {lang_name} ({lang}): {count} samples")

    has_question = sum(1 for x in dataset if x.get('question'))
    has_answer = sum(1 for x in dataset if x.get('answer'))
    has_options = sum(1 for x in dataset if x.get('options') and len(x['options']) > 0)

    print(f"\nData completeness:")
    print(f"  With question: {has_question}/{len(dataset)} ({has_question/len(dataset)*100:.1f}%)")
    print(f"  With answer: {has_answer}/{len(dataset)} ({has_answer/len(dataset)*100:.1f}%)")
    print(f"  With options: {has_options}/{len(dataset)} ({has_options/len(dataset)*100:.1f}%)")

    print(f"\nSample data ({num_samples} examples):")
    print("-" * 70)

    shown = 0
    for sample in dataset:
        if shown >= num_samples:
            break
        if not sample.get('options') or not sample.get('answer'):
            continue

        lang = sample.get('language', 'unknown')
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1]

        print(f"\n[{shown + 1}] {lang_name} ({lang})")
        print(f"  Source: {sample.get('source_language', 'unknown')}")
        print(f"  Question: {str(sample.get('question', ''))[:80]}...")
        print(f"  Answer: {sample.get('answer')}")
        print(f"  Options: {len(sample.get('options', []))}")

        shown += 1

    print("\n" + "=" * 70)
    response = input("\nProceed with experiment? (y/n): ")
    return response.lower() == 'y'


def validate_sample(sample: Dict) -> Optional[str]:
    if not sample.get('question'):
        return "Missing question"

    if not sample.get('answer'):
        return "Missing answer"

    options = sample.get('options')
    if not options or not isinstance(options, list) or len(options) == 0:
        return "Missing or empty options"

    answer = sample['answer']
    if isinstance(answer, list):
        if not answer:
            return "Empty answer array"
        answer = str(answer[0]).strip().upper()
    else:
        answer = str(answer).strip().upper()

    if len(answer) != 1 or not answer.isalpha():
        return f"Answer '{sample['answer']}' is not a single letter"

    answer_index = ord(answer) - ord('A')
    if answer_index < 0 or answer_index >= len(options):
        return f"Answer '{answer}' invalid for {len(options)} options"

    return None


def format_mcq(question: str, options: List[str]) -> str:
    formatted = f"{question}\n\nOptions:\n"
    for i, opt in enumerate(options):
        formatted += f"{chr(65 + i)}. {opt}\n"
    formatted += "\nProvide only the letter of the correct answer:"
    return formatted


def generate_answer(model, tokenizer, collator, question: str, options: List[str]) -> str:
    formatted_q = format_mcq(question, options)
    sample = {
        "conversations": [{"role": "user", "content": formatted_q}],
        "modalities": []
    }

    batch = collator([sample])
    batch = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            batch=batch,
            temperature=0.1,
            do_sample=False,
            max_new_tokens=10
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class MedicalQAEvaluator:
    def __init__(self):
        self.results = {'translation': [], 'native': []}
        self.skipped_samples = []
        self.fasttext_stats = {
            'total': 0,
            'correct': 0,
            'by_language': {},
            'disagreements': []
        }

    def track_fasttext(self, true_lang: str, detected_lang: str, confidence: float, question: str):
        self.fasttext_stats['total'] += 1

        if true_lang not in self.fasttext_stats['by_language']:
            self.fasttext_stats['by_language'][true_lang] = {
                'total': 0,
                'correct': 0,
                'wrong_detections': Counter()
            }

        self.fasttext_stats['by_language'][true_lang]['total'] += 1

        if detected_lang == true_lang:
            self.fasttext_stats['correct'] += 1
            self.fasttext_stats['by_language'][true_lang]['correct'] += 1
        else:
            self.fasttext_stats['by_language'][true_lang]['wrong_detections'][detected_lang] += 1
            self.fasttext_stats['disagreements'].append({
                'true': true_lang,
                'detected': detected_lang,
                'confidence': confidence,
                'question': question[:60] + '...'
            })

    def extract_answer_letter(self, text: str, num_options: int) -> str:
        text = text.upper().strip()
        valid_letters = {chr(65 + i) for i in range(num_options)}

        patterns = [
            r'\bANSWER\s*:?\s*([A-Z])\b',
            r'\bTHE ANSWER IS\s*:?\s*([A-Z])\b',
            r'\b([A-Z])\s*IS\s+(?:THE\s+)?CORRECT',
            r'\bOPTION\s*([A-Z])\b',
            r'\bCHOICE\s*([A-Z])\b',
            r'^([A-Z])\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match and match.group(1) in valid_letters:
                return match.group(1)

        match = re.search(r'\b([A-Z])\b', text)
        if match and match.group(1) in valid_letters:
            return match.group(1)

        for letter in sorted(valid_letters):
            if letter in text:
                return letter

        return "UNKNOWN"

    def add_result(self, pipeline: str, question: str, language: str,
                   correct: str, predicted: str, num_options: int,
                   detected_lang: str = None, confidence: float = None):
        self.results[pipeline].append({
            'question': question[:100] + '...',
            'language': language,
            'correct': str(correct).upper(),
            'predicted': str(predicted).upper(),
            'is_correct': str(predicted).upper() == str(correct).upper(),
            'num_options': num_options,
            'detected_lang': detected_lang,
            'confidence': confidence
        })

    def add_skipped(self, reason: str, language: str = None):
        self.skipped_samples.append({'reason': reason, 'language': language})

    def compute_stats(self) -> Dict:
        stats = {}

        for pipeline in ['translation', 'native']:
            results = self.results[pipeline]
            if not results:
                stats[pipeline] = {'accuracy': 0, 'total': 0, 'correct': 0, 'by_language': {}}
                continue

            total = len(results)
            correct = sum(1 for r in results if r['is_correct'])

            by_lang = {}
            for r in results:
                lang = r['language']
                if lang not in by_lang:
                    by_lang[lang] = {'correct': 0, 'total': 0}
                by_lang[lang]['total'] += 1
                if r['is_correct']:
                    by_lang[lang]['correct'] += 1

            for lang in by_lang:
                by_lang[lang]['accuracy'] = by_lang[lang]['correct'] / by_lang[lang]['total'] * 100

            stats[pipeline] = {
                'accuracy': correct / total * 100,
                'correct': correct,
                'total': total,
                'by_language': by_lang
            }

        skip_reasons = Counter(s['reason'] for s in self.skipped_samples)
        skip_by_lang = Counter(s['language'] for s in self.skipped_samples if s['language'])
        stats['skipped'] = {
            'total': len(self.skipped_samples),
            'by_reason': dict(skip_reasons),
            'by_language': dict(skip_by_lang)
        }

        return stats

    def print_summary(self):
        stats = self.compute_stats()

        print("\n" + "=" * 70)
        print("AFRICAN LANGUAGES: TRANSLATION VS NATIVE MULTILINGUAL (BASE NLLB)")
        print("=" * 70)

        ft = self.fasttext_stats
        if ft['total'] > 0:
            accuracy = ft['correct'] / ft['total'] * 100
            print(f"\nFastText Detection Accuracy: {accuracy:.1f}% ({ft['correct']}/{ft['total']})")

            print("\nBy Language:")
            for lang, data in sorted(ft['by_language'].items(), key=lambda x: -x[1]['total']):
                acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
                status = "OK" if acc > 80 else "WARN" if acc > 50 else "FAIL"
                print(f"  [{status}] {lang:10s}: {acc:5.1f}% ({data['correct']:4d}/{data['total']:4d})", end='')
                if data['wrong_detections']:
                    top_wrong = data['wrong_detections'].most_common(2)
                    wrong_str = ', '.join([f"{k}({v})" for k, v in top_wrong])
                    print(f"  -> confused with: {wrong_str}")
                else:
                    print()

        if stats['skipped']['total'] > 0:
            print(f"\nSkipped Samples: {stats['skipped']['total']}")
            for reason, count in sorted(stats['skipped']['by_reason'].items(), key=lambda x: -x[1])[:5]:
                print(f"  {reason}: {count}")

        for pipeline, label in [('translation', 'TRANSLATION PIPELINE'), ('native', 'NATIVE MULTILINGUAL')]:
            data = stats[pipeline]
            print(f"\n{label}:")
            print(f"  Overall Accuracy: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")
            if data.get('by_language'):
                print("\n  By language:")
                for lang, lang_data in sorted(data['by_language'].items(), key=lambda x: -x[1]['total']):
                    lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1]
                    print(f"    {lang_name:12s} ({lang}): {lang_data['accuracy']:5.1f}% "
                          f"({lang_data['correct']:3d}/{lang_data['total']:3d})")

        trans = stats['translation']
        native = stats['native']
        if trans['total'] > 0 and native['total'] > 0:
            diff = trans['accuracy'] - native['accuracy']
            print(f"\nCOMPARISON:")
            print(f"  Overall Difference: {diff:+.2f} percentage points")
            print("\n  Per-language (Translation - Native):")

            all_langs = set(trans['by_language'].keys()) | set(native['by_language'].keys())
            for lang in sorted(all_langs):
                trans_acc = trans['by_language'].get(lang, {}).get('accuracy', 0)
                native_acc = native['by_language'].get(lang, {}).get('accuracy', 0)
                lang_diff = trans_acc - native_acc
                count = trans['by_language'].get(lang, {}).get('total', 0)
                if count > 0:
                    lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1]
                    status = "+" if lang_diff > 2 else "-" if lang_diff < -2 else "="
                    print(f"    [{status}] {lang_name:12s}: {lang_diff:+6.1f}% (n={count})")

            print(f"\nCONCLUSION:")
            if abs(diff) < 2:
                print("  Translation does not significantly impact performance")
            elif diff > 2:
                print(f"  Translation HELPS (+{diff:.2f}%) - translation system is justified")
            else:
                print(f"  Translation HURTS ({diff:.2f}%) - focus on multilingual capability")

        print("=" * 70)

    def save_results(self, output_path: str):
        output = {
            'experiment': 'African Languages: Translation vs Native Multilingual (Base NLLB)',
            'statistics': self.compute_stats(),
            'fasttext_performance': self.fasttext_stats,
            'detailed_results': self.results,
            'skipped_samples': self.skipped_samples
        }

        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")


def run_experiment(args):
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading translated dataset...")
    dataset = load_translated_dataset(args.input)

    if not args.skip_exploration:
        if not explore_dataset(dataset, num_samples=5):
            print("\nExperiment cancelled")
            return

    print("\n[2/5] Loading models...")

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        padding_side="left",
        token=os.getenv("HF_PERSONAL_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': [ATTACHMENT_TOKEN]})

    model = MultiModalModelForCausalLM.from_pretrained(
        "ClosedMeditron/Mulimeditron-Proj-CLIP-generalist",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    model.to("cuda")
    model.eval()

    translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")

    chat_template = ChatTemplate.from_name("llama")
    collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        modality_processors=model.processors(),
        modality_loaders={},
        attachment_token=ATTACHMENT_TOKEN,
        chat_template=chat_template,
        add_generation_prompt=True,
    )

    evaluator = MedicalQAEvaluator()

    print("\n[3/5] Validating and stratifying samples...")

    samples_by_lang: Dict[str, List[Dict]] = {}
    for sample in tqdm(dataset, desc="  Validating"):
        error = validate_sample(sample)
        if error:
            evaluator.add_skipped(error, sample.get('language'))
            continue
        lang = sample.get('language', 'unknown')
        samples_by_lang.setdefault(lang, []).append(sample)

    print("\nValid samples by language:")
    for lang, samples in sorted(samples_by_lang.items(), key=lambda x: -len(x[1])):
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1]
        print(f"  {lang_name} ({lang}): {len(samples)}")

    valid_samples: List[Dict] = []
    for lang, samples in samples_by_lang.items():
        take = min(args.samples_per_language, len(samples))
        valid_samples.extend(samples[:take])
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1]
        print(f"  Selected {take} from {lang_name}")

    print(f"\nTotal samples for experiment: {len(valid_samples)}")
    if len(valid_samples) == 0:
        print("No valid samples found")
        return

    print("\n[4/5] Running evaluation...")
    samples_since_cleanup = 0

    for idx in tqdm(range(len(valid_samples)), desc="  Processing"):
        sample = valid_samples[idx]
        question = sample['question']
        correct_answer = sample['answer']
        options = sample['options']
        language = sample.get('language', 'unknown')
        num_options = len(options)

        try:
            nllb_code, _ = AFRICAN_LANGUAGES.get(language, ('eng_Latn', 'Unknown'))

            detected_lang = translator.detect_language(question, confidence_threshold=0.3)
            confidence = float(translator.lang_detector.predict(question, k=1)[1][0])
            evaluator.track_fasttext(nllb_code, detected_lang, confidence, question)

            english_question = translator.translate(question, src_lang=nllb_code, tgt_lang='eng_Latn')
            english_options = [
                translator.translate(opt, src_lang=nllb_code, tgt_lang='eng_Latn')
                for opt in options
            ]

            response_trans = generate_answer(model, tokenizer, collator, english_question, english_options)
            predicted_trans = evaluator.extract_answer_letter(response_trans, num_options)

            evaluator.add_result(
                pipeline='translation',
                question=question,
                language=language,
                correct=correct_answer,
                predicted=predicted_trans,
                num_options=num_options,
                detected_lang=detected_lang,
                confidence=confidence
            )

            response_native = generate_answer(model, tokenizer, collator, question, options)
            predicted_native = evaluator.extract_answer_letter(response_native, num_options)

            evaluator.add_result(
                pipeline='native',
                question=question,
                language=language,
                correct=correct_answer,
                predicted=predicted_native,
                num_options=num_options
            )

            samples_since_cleanup += 1
            if samples_since_cleanup >= 25:
                torch.cuda.empty_cache()
                gc.collect()
                samples_since_cleanup = 0

            if (idx + 1) % 25 == 0:
                stats = evaluator.compute_stats()
                trans_acc = stats['translation']['accuracy']
                native_acc = stats['native']['accuracy']
                print(f"\n  [{idx+1}/{len(valid_samples)}] "
                      f"Trans: {trans_acc:.1f}% | Native: {native_acc:.1f}% | "
                      f"Diff: {trans_acc - native_acc:+.1f}%")

        except Exception as e:
            print(f"\n  Error at sample {idx+1}: {e}")
            evaluator.add_skipped(f"Generation error: {str(e)[:80]}", language)

    print("\n[5/5] Computing and saving results...")
    evaluator.print_summary()
    evaluator.save_results(args.output)

    print("\nExperiment complete.\n")


def main():
    parser = argparse.ArgumentParser(
        description="African Languages: Translation vs Native Multilingual (Base NLLB)"
    )
    parser.add_argument(
        "--input",
        default="src/multimeditron/translation/experiments/results/base_nllb/mediBench_translation/cleaned_high_resource_translation_results.json",
        help="Path to translated JSON file"
    )
    parser.add_argument(
        "--samples_per_language",
        type=int,
        default=1000,
        help="Samples per language (default: 1000)"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_3_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--skip_exploration",
        action="store_true",
        help="Skip dataset exploration prompt"
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
