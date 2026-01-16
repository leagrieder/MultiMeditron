"""
African Languages Experiment Using Pre-Translated Dataset

Compares translation pipeline vs native multilingual for African languages:
1. Translation pipeline (NLLB → Meditron)
2. Native multilingual (Direct multilingual Meditron)

Uses pre-translated medical MCQ dataset in African languages (Amharic, Hausa, Swahili, Yoruba, Zulu).

Default: 1000 samples per language

Output: src/multimeditron/translation/experiments/results/base_nllb/experiment_3_results.json
"""

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import re
import os
import argparse
import sys
from typing import List, Dict, Optional
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimeditron.model.model import MultiModalModelForCausalLM
from multimeditron.dataset.preprocessor.modality_preprocessor import ModalityRetriever
from multimeditron.dataset.registry.fs_registry import FileSystemImageRegistry
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.translation.translator import NLLBTranslator


AFRICAN_LANGUAGES = {
    'am': ('amh_Ethi', 'Amharic'),
    'ha': ('hau_Latn', 'Hausa'),
    'sw': ('swh_Latn', 'Swahili'),
    'yo': ('yor_Latn', 'Yoruba'),
    'zu': ('zul_Latn', 'Zulu'),
}


def load_translated_dataset(json_path: str):
    """Load translated African languages dataset from JSON."""
    print(f"Loading translated dataset from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Loaded {len(data)} translated MCQs")
    
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __iter__(self):
            return iter(self.data)
    
    return SimpleDataset(data)


def explore_dataset(dataset, num_samples=5):
    """Show dataset structure and get user confirmation."""
    print("\n" + "="*70)
    print("AFRICAN LANGUAGES DATASET EXPLORATION")
    print("="*70)
    
    print(f"\nTotal samples: {len(dataset)}")
    
    languages = Counter(sample.get('language') for sample in dataset)
    print(f"\nLanguage distribution:")
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1] if lang in AFRICAN_LANGUAGES else lang
        print(f"  {lang_name} ({lang}): {count} samples")
    
    has_question = sum(1 for x in dataset if x.get('question'))
    has_answer = sum(1 for x in dataset if x.get('answer'))
    has_options = sum(1 for x in dataset if x.get('options') and len(x['options']) > 0)
    
    print(f"\nData completeness:")
    print(f"  Samples with question: {has_question}/{len(dataset)} ({has_question/len(dataset)*100:.1f}%)")
    print(f"  Samples with answer: {has_answer}/{len(dataset)} ({has_answer/len(dataset)*100:.1f}%)")
    print(f"  Samples with options: {has_options}/{len(dataset)} ({has_options/len(dataset)*100:.1f}%)")
    
    option_counts = Counter()
    for sample in dataset:
        opts = sample.get('options')
        if opts and isinstance(opts, list):
            option_counts[len(opts)] += 1
    
    print(f"\nNumber of options per question:")
    for num_opts, count in sorted(option_counts.items()):
        print(f"  {num_opts} options: {count} samples")
    
    print(f"\n{'='*70}")
    print(f"SAMPLE DATA (first {num_samples} with options):")
    print(f"{'='*70}")
    
    samples_shown = 0
    for sample in dataset:
        if samples_shown >= num_samples:
            break
        
        if not sample.get('options') or not sample.get('answer'):
            continue
        
        lang = sample.get('language', 'unknown')
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1] if lang in AFRICAN_LANGUAGES else lang
        
        print(f"\nSample {samples_shown + 1}:")
        print(f"  Language: {lang_name} ({lang})")
        print(f"  Source: {sample.get('source_language', 'unknown')}")
        print(f"  Question: {str(sample['question'])[:80]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Options ({len(sample['options'])}):")
        for j, opt in enumerate(sample['options'][:5]):
            print(f"    {chr(65+j)}. {str(opt)[:60]}...")
        
        samples_shown += 1
    
    print(f"\n{'='*70}")
    
    response = input("\n❓ Proceed with experiment? (y/n): ")
    return response.lower() == 'y'


def validate_sample(sample: Dict) -> Optional[str]:
    """Return error message if sample is invalid, None if valid."""
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


def format_question_with_options(question: str, options: List[str]) -> str:
    """Format multiple choice question."""
    formatted = f"{question}\n\nOptions:\n"
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        formatted += f"{letter}. {opt}\n"
    formatted += "\nProvide only the letter of the correct answer:"
    return formatted


def generate_answer(model, tokenizer, collator, modality_retriever, 
                   question: str, options: List[str]) -> str:
    """Generate Meditron's answer to a multiple choice question."""
    formatted_q = format_question_with_options(question, options)
    
    conversations = [{"role": "user", "content": formatted_q}]
    sample = {"conversations": conversations, "modalities": []}
    sample = modality_retriever.merge_modality_with_sample(sample)
    batch = collator([sample])
    
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            batch=batch,
            temperature=0.1,
            do_sample=False,
            max_new_tokens=10
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


class MedicalQAEvaluator:
    """Tracks and reports accuracy for both translation and native pipelines."""
    
    def __init__(self):
        self.results = {
            'translation': [],
            'native': []
        }
        self.skipped_samples = []
        self.fasttext_stats = {
            'total': 0,
            'correct': 0,
            'by_language': {},
            'disagreements': []
        }
    
    def track_fasttext(self, true_lang: str, detected_lang: str, confidence: float, question: str):
        """Compare fastText detection against ground truth."""
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
        """Extract answer letter from model output, constrained by number of options."""
        text = text.upper().strip()
        valid_letters = set(chr(65 + i) for i in range(num_options))
        
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
        """Store a result for a pipeline."""
        self.results[pipeline].append({
            'question': question[:100] + '...',
            'language': language,
            'correct': correct.upper(),
            'predicted': predicted.upper(),
            'is_correct': predicted.upper() == correct.upper(),
            'num_options': num_options,
            'detected_lang': detected_lang,
            'confidence': confidence
        })
    
    def add_skipped(self, reason: str, language: str = None):
        """Track skipped sample."""
        self.skipped_samples.append({
            'reason': reason,
            'language': language
        })
    
    def compute_stats(self) -> Dict:
        """Compute accuracy by pipeline and language."""
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
        """Print comparison of translation vs native pipelines."""
        stats = self.compute_stats()
        
        print("\n" + "="*70)
        print("EXPERIMENT: AFRICAN LANGUAGES - DOES MEDITRON NEED TRANSLATION?")
        print("="*70)
        
        ft = self.fasttext_stats
        if ft['total'] > 0:
            accuracy = ft['correct'] / ft['total'] * 100
            print(f"\n📍 FASTTEXT DETECTION ACCURACY: {accuracy:.1f}% ({ft['correct']}/{ft['total']})")
            
            print(f"\n   By True Language:")
            sorted_langs = sorted(ft['by_language'].items(), key=lambda x: -x[1]['total'])
            for lang, data in sorted_langs:
                acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
                symbol = "✅" if acc > 80 else "⚠️" if acc > 50 else "❌"
                print(f"      {symbol} {lang:10s}: {acc:5.1f}% ({data['correct']:4d}/{data['total']:4d})", end='')
                
                if data['wrong_detections']:
                    top_wrong = data['wrong_detections'].most_common(2)
                    wrong_str = ', '.join([f"{k}({v})" for k, v in top_wrong])
                    print(f"  → confused with: {wrong_str}")
                else:
                    print()
            
            if ft['disagreements'][:3]:
                print(f"\n   Sample Disagreements:")
                for i, d in enumerate(ft['disagreements'][:3], 1):
                    print(f"      {i}. True:{d['true']:10s} Detected:{d['detected']:10s} Conf:{d['confidence']:.2f}")
                    print(f"         \"{d['question']}\"")
        
        if stats['skipped']['total'] > 0:
            print(f"\n⚠️  SKIPPED SAMPLES: {stats['skipped']['total']}")
            for reason, count in sorted(stats['skipped']['by_reason'].items(), key=lambda x: -x[1])[:5]:
                print(f"      {reason}: {count}")
        
        trans = stats['translation']
        native = stats['native']
        
        print(f"\n🔄 TRANSLATION PIPELINE:")
        print(f"   Overall Accuracy: {trans['accuracy']:.2f}% ({trans['correct']}/{trans['total']})")
        if trans.get('by_language'):
            print(f"\n   By language:")
            for lang, data in sorted(trans['by_language'].items(), key=lambda x: -x[1]['total']):
                lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1] if lang in AFRICAN_LANGUAGES else lang
                print(f"      {lang_name:12s} ({lang}): {data['accuracy']:5.1f}% ({data['correct']:3d}/{data['total']:3d})")
        
        print(f"\n🌍 NATIVE MULTILINGUAL PIPELINE:")
        print(f"   Overall Accuracy: {native['accuracy']:.2f}% ({native['correct']}/{native['total']})")
        if native.get('by_language'):
            print(f"\n   By language:")
            for lang, data in sorted(native['by_language'].items(), key=lambda x: -x[1]['total']):
                lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1] if lang in AFRICAN_LANGUAGES else lang
                print(f"      {lang_name:12s} ({lang}): {data['accuracy']:5.1f}% ({data['correct']:3d}/{data['total']:3d})")
        
        if trans['total'] > 0 and native['total'] > 0:
            diff = trans['accuracy'] - native['accuracy']
            print(f"\n📊 OVERALL COMPARISON:")
            print(f"   Difference: {diff:+.2f} percentage points")
            
            print(f"\n   Per-language differences (Translation - Native):")
            all_langs = set(trans['by_language'].keys()) | set(native['by_language'].keys())
            lang_diffs = []
            for lang in all_langs:
                trans_acc = trans['by_language'].get(lang, {}).get('accuracy', 0)
                native_acc = native['by_language'].get(lang, {}).get('accuracy', 0)
                lang_diff = trans_acc - native_acc
                lang_count = trans['by_language'].get(lang, {}).get('total', 0)
                lang_diffs.append((lang, lang_diff, lang_count))
            
            for lang, lang_diff, count in sorted(lang_diffs, key=lambda x: -x[2]):
                if count > 0:
                    lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1] if lang in AFRICAN_LANGUAGES else lang
                    symbol = "✅" if lang_diff > 2 else "⚠️" if lang_diff < -2 else "➡️"
                    print(f"      {symbol} {lang_name:12s}: {lang_diff:+6.1f}% (n={count})")
            
            print(f"\n💡 CONCLUSION:")
            if abs(diff) < 2:
                print(f"   ➡️  Translation doesn't significantly impact performance")
            elif diff > 2:
                print(f"   ✅ Translation HELPS! (+{diff:.2f}%)")
                print(f"      → Building a translation system is JUSTIFIED")
            else:
                print(f"   ⚠️  Translation HURTS! ({diff:.2f}%)")
                print(f"      → Focus on multilingual capability instead")
        
        print("="*70)
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        output = {
            'experiment': 'African Languages: Does Meditron Need Translation?',
            'statistics': self.compute_stats(),
            'fasttext_performance': self.fasttext_stats,
            'detailed_results': self.results,
            'skipped_samples': self.skipped_samples
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_path}")


def run_experiment(args):
    """Run African languages experiment with pre-translated dataset."""
    print("\n[1/6] Loading translated African languages dataset...")
    dataset = load_translated_dataset(args.input)
    print(f"   Loaded {len(dataset)} samples")
    
    if not args.skip_exploration:
        if not explore_dataset(dataset, num_samples=5):
            print("\n❌ Experiment cancelled by user")
            return
    
    print("\n[2/6] Loading models...")
    
    default_llm = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        default_llm,
        padding_side="left",
        token=os.getenv("HF_PERSONAL_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
    tokenizer.add_special_tokens({'additional_special_tokens': [ATTACHMENT_TOKEN]})
    attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)
    
    model = MultiModalModelForCausalLM.from_pretrained(
        "ClosedMeditron/Mulimeditron-Proj-CLIP-generalist",
        dtype=torch.bfloat16,
        use_safetensors=True,
        token=os.getenv("HF_LAB_TOKEN"),
        llm_token=os.getenv("HF_PERSONAL_TOKEN")
    )
    model.to("cuda")
    model.eval()
    
    # ✅ VERIFIED: Using finetuned model
    translator = NLLBTranslator()
    
    modality_retriever = ModalityRetriever(
        registry=FileSystemImageRegistry(base_path=os.getcwd())
    )
    collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        tokenizer_type="llama",
        modality_processors=model.processors(),
        attachment_token_idx=attachment_token_idx,
        add_generation_prompt=True,
    )
    
    evaluator = MedicalQAEvaluator()
    
    print("\n[3/6] Validating samples and stratifying by language...")

    samples_by_lang = {}
    for sample in tqdm(dataset, desc="Validating"):
        error = validate_sample(sample)
        if error:
            evaluator.add_skipped(error, sample.get('language'))
        else:
            lang = sample.get('language', 'unknown')
            if lang not in samples_by_lang:
                samples_by_lang[lang] = []
            samples_by_lang[lang].append(sample)

    print(f"\n   Valid samples by language:")
    for lang, samples in sorted(samples_by_lang.items(), key=lambda x: -len(x[1])):
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1] if lang in AFRICAN_LANGUAGES else lang
        print(f"      {lang_name} ({lang}): {len(samples)} samples")

    SAMPLES_PER_LANGUAGE = args.samples_per_language
    valid_samples = []
    for lang, samples in samples_by_lang.items():
        take = min(SAMPLES_PER_LANGUAGE, len(samples))
        valid_samples.extend(samples[:take])
        lang_name = AFRICAN_LANGUAGES.get(lang, (None, lang))[1] if lang in AFRICAN_LANGUAGES else lang
        print(f"   Selected {take} samples from {lang_name} ({lang})")

    print(f"\n   Total samples for experiment: {len(valid_samples)}")

    if len(valid_samples) == 0:
        print("❌ No valid samples found!")
        return
    
    print(f"\n[4/6] Running both pipelines on {len(valid_samples)} samples...")
    print(f"   This will take approximately {len(valid_samples) * 2 * 3 / 60:.0f} minutes")
    
    for idx, sample in enumerate(tqdm(valid_samples, desc="Processing")):
        question = sample['question']
        correct_answer = sample['answer']
        options = sample['options']
        language = sample.get('language', 'unknown')
        num_options = len(options)
        
        try:
            nllb_code, lang_name = AFRICAN_LANGUAGES.get(language, ('eng_Latn', 'Unknown'))
            
            detected_lang_ft = translator.detect_language(question, confidence_threshold=0.3)
            confidence = float(translator.lang_detector.predict(question, k=1)[1][0])
            evaluator.track_fasttext(nllb_code, detected_lang_ft, confidence, question)
            
            # Translation pipeline
            english_question = translator.translate(question, src_lang=nllb_code, tgt_lang='eng_Latn')
            english_options = [
                translator.translate(opt, src_lang=nllb_code, tgt_lang='eng_Latn')
                for opt in options
            ]
            response_trans = generate_answer(
                model, tokenizer, collator, modality_retriever,
                english_question, english_options
            )
            predicted_trans = evaluator.extract_answer_letter(response_trans, num_options)
            
            evaluator.add_result(
                pipeline='translation',
                question=question,
                language=language,
                correct=correct_answer,
                predicted=predicted_trans,
                num_options=num_options,
                detected_lang=detected_lang_ft,
                confidence=confidence
            )
            
            # Native pipeline
            response_native = generate_answer(
                model, tokenizer, collator, modality_retriever,
                question, options
            )
            predicted_native = evaluator.extract_answer_letter(response_native, num_options)
            
            evaluator.add_result(
                pipeline='native',
                question=question,
                language=language,
                correct=correct_answer,
                predicted=predicted_native,
                num_options=num_options
            )
            
            if (idx + 1) % 25 == 0:
                temp_stats = evaluator.compute_stats()
                trans_acc = temp_stats['translation']['accuracy']
                native_acc = temp_stats['native']['accuracy']
                print(f"\n[Progress at {idx+1}/{len(valid_samples)}]")
                print(f"  Translation: {trans_acc:.1f}% | Native: {native_acc:.1f}% | Diff: {trans_acc-native_acc:+.1f}%")
        
        except Exception as e:
            print(f"\n[ERROR] Sample {idx+1} failed: {e}")
            evaluator.add_skipped(f"Generation error: {str(e)[:50]}", language)
            continue
    
    print("\n[5/6] Computing results...")
    evaluator.print_summary()
    
    print("\n[6/6] Saving results...")
    evaluator.save_results(args.output)
    
    print("\n✅ African Languages Experiment complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="African Languages: Does Meditron Need Translation?"
    )
    parser.add_argument(
        "--input",
        default="src/multimeditron/translation/experiments/results/finetuned_nllb/mediBench_translation/cleaned_high_resource_translation_results_subset.json",
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
        default="src/multimeditron/translation/experiments/results/finetuned_nllb/experiment_3_results.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--skip_exploration",
        action="store_true",
        help="Skip dataset exploration step"
    )
    
    args = parser.parse_args()
    run_experiment(args)