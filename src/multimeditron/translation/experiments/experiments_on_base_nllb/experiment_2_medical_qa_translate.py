"""
High-Resource Language Translation Experiment

Compares Meditron's native translation vs NLLB for high-resource languages on medical Q&A.
Uses MediBench dataset (no reference translations available).

Evaluation approach:
- NLLB translations as pseudo-references (BLEU/chrF)
- BERTScore for semantic similarity
- Medical term preservation analysis

Runs in two passes to avoid GPU OOM: Meditron first, then NLLB.

Default: 1000 samples per language

Output: src/multimeditron/translation/experiments/results/base_nllb/experiment_2_results.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import sys
import re
from pathlib import Path
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from collections import defaultdict, Counter
import evaluate
import fasttext
from huggingface_hub import hf_hub_download
from bert_score import score as bert_score

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multimeditron.model.model import MultiModalModelForCausalLM
from multimeditron.dataset.preprocessor.modality_preprocessor import ModalityRetriever
from multimeditron.dataset.registry.fs_registry import FileSystemImageRegistry
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.translation.translator import NLLBTranslator


HIGH_RESOURCE_LANGS = {
    'zh': ('zho_Hans', 'Chinese (Simplified)'),
    'es': ('spa_Latn', 'Spanish'),
    'fr': ('fra_Latn', 'French'),
    'ja': ('jpn_Jpan', 'Japanese'),
    'ru': ('rus_Cyrl', 'Russian'),
}


def load_medibench_robust(split="train", token=None):
    """Load MediBench with handling for inconsistent data format."""
    print(f"   Loading MediBench {split} split with custom loader...")
    
    file_path = hf_hub_download(
        repo_id="ClosedMeditron/MediBench",
        filename=f"{split}.jsonl",
        repo_type="dataset",
        token=token
    )
    
    samples = []
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                sample = json.loads(line.strip())
                
                if isinstance(sample.get('answer'), list):
                    sample['answer'] = sample['answer'][0] if sample['answer'] else None
                
                if sample.get('answer') is not None:
                    sample['answer'] = str(sample['answer'])
                
                for field in ['question', 'rationale', 'meta_info', 'subject', 'equation_solution']:
                    if isinstance(sample.get(field), list):
                        sample[field] = ' '.join(str(x) for x in sample[field]) if sample[field] else None
                
                for field in ['options', 'choices', 'metamap_phrases']:
                    if sample.get(field) is None:
                        sample[field] = []
                    elif not isinstance(sample.get(field), list):
                        sample[field] = [str(sample[field])]
                
                samples.append(sample)
                
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"      ⚠️  Skipping malformed line {line_num}: {str(e)[:60]}")
                continue
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"      ⚠️  Error on line {line_num}: {str(e)[:60]}")
                continue
    
    if skipped > 5:
        print(f"      ⚠️  ... and {skipped - 5} more skipped lines")
    
    print(f"   ✅ Successfully loaded {len(samples)} samples")
    return Dataset.from_list(samples)


class HighResourceEvaluator:
    """Evaluates translation quality using multiple metrics with NLLB as pseudo-reference."""
    
    def __init__(self):
        self.results = {
            'meditron_direct': defaultdict(list),
            'nllb_baseline': defaultdict(list)
        }
        
        self.detection_stats = {
            'total': 0,
            'correct': 0,
            'by_language': defaultdict(lambda: {'total': 0, 'correct': 0, 'misdetections': Counter()})
        }
        
        self.bleu = evaluate.load('bleu')
        self.chrf = evaluate.load('chrf')
        print("   📊 Loading BERTScore...")
        
    def track_detection(self, true_lang: str, detected_lang: str):
        """Track fastText detection against ground truth."""
        self.detection_stats['total'] += 1
        self.detection_stats['by_language'][true_lang]['total'] += 1
        
        if detected_lang == true_lang:
            self.detection_stats['correct'] += 1
            self.detection_stats['by_language'][true_lang]['correct'] += 1
        else:
            self.detection_stats['by_language'][true_lang]['misdetections'][detected_lang] += 1
    
    def add_result(self, pipeline: str, language: str,
                   source: str, prediction: str, 
                   nllb_reference: str = None,
                   detected_lang: str = None):
        """Store translation result."""
        self.results[pipeline][language].append({
            'source': source[:200],
            'prediction': prediction,
            'nllb_reference': nllb_reference,
            'detected_lang': detected_lang
        })
    
    def compute_metrics(self):
        """Compute BLEU, chrF, and BERTScore comparing Meditron to NLLB."""
        stats = {}
        
        for language, translations in self.results['meditron_direct'].items():
            if not translations:
                continue
            
            meditron_preds = [t['prediction'] for t in translations]
            nllb_refs = [[t['nllb_reference']] for t in translations]
            
            bleu_result = self.bleu.compute(predictions=meditron_preds, references=nllb_refs)
            chrf_result = self.chrf.compute(predictions=meditron_preds, references=nllb_refs)
            
            print(f"\n   Computing BERTScore for {language}...")
            nllb_flat = [t['nllb_reference'] for t in translations]
            
            P, R, F1 = bert_score(
                meditron_preds, 
                nllb_flat,
                lang='en',
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            bert_f1 = F1.mean().item() * 100
            
            stats[language] = {
                'count': len(translations),
                'bleu': bleu_result['bleu'] * 100,
                'chrf': chrf_result['score'],
                'bertscore_f1': bert_f1
            }
        
        return stats
    
    def analyze_medical_terms(self):
        """Check if medical terminology is preserved in translations."""
        medical_indicators = [
            'patient', 'treatment', 'diagnosis', 'symptom', 'therapy',
            'disease', 'syndrome', 'disorder', 'medication', 'surgery',
            'clinical', 'medical', 'health', 'test', 'exam'
        ]
        
        preservation_stats = {}
        
        for language, translations in self.results['meditron_direct'].items():
            preserved_count = 0
            total_count = 0
            
            for t in translations:
                nllb_lower = t['nllb_reference'].lower()
                meditron_lower = t['prediction'].lower()
                
                nllb_has_medical = any(term in nllb_lower for term in medical_indicators)
                
                if nllb_has_medical:
                    total_count += 1
                    meditron_has_medical = any(term in meditron_lower for term in medical_indicators)
                    if meditron_has_medical:
                        preserved_count += 1
            
            if total_count > 0:
                preservation_stats[language] = {
                    'preserved': preserved_count,
                    'total': total_count,
                    'rate': preserved_count / total_count * 100
                }
        
        return preservation_stats
    
    def print_summary(self):
        """Print comprehensive comparison of Meditron vs NLLB."""
        stats = self.compute_metrics()
        med_stats = self.analyze_medical_terms()
        
        print("\n" + "="*80)
        print("EXPERIMENT: HIGH-RESOURCE LANGUAGE TRANSLATION EVALUATION")
        print("Comparing Meditron vs NLLB on Medical Q&A")
        print("="*80)
        
        det = self.detection_stats
        if det['total'] > 0:
            det_acc = det['correct'] / det['total'] * 100
            print(f"\n📍 FASTTEXT DETECTION: {det_acc:.1f}% accuracy ({det['correct']}/{det['total']})")
        
        print("\n" + "="*80)
        print("📊 TRANSLATION QUALITY (Meditron vs NLLB Baseline)")
        print("="*80)
        print(f"\n{'Language':<15} {'BLEU':<8} {'chrF':<8} {'BERTScore':<10} {'Med Terms':<12} {'Assessment'}")
        print("-" * 80)
        
        winners = {'meditron': 0, 'nllb': 0, 'tie': 0}
        
        for lang in sorted(stats.keys()):
            s = stats[lang]
            med = med_stats.get(lang, {'rate': 0})
            
            if s['bertscore_f1'] >= 90:
                assessment = "✅ Excellent"
                winners['meditron'] += 1
            elif s['bertscore_f1'] >= 80:
                assessment = "✅ Good"
                winners['meditron'] += 1
            elif s['bertscore_f1'] >= 70:
                assessment = "⚠️  Acceptable"
                winners['tie'] += 1
            else:
                assessment = "❌ Poor"
                winners['nllb'] += 1
            
            print(f"{HIGH_RESOURCE_LANGS[lang][1]:<15} "
                  f"{s['bleu']:>6.1f}% "
                  f"{s['chrf']:>6.1f}  "
                  f"{s['bertscore_f1']:>8.1f}  "
                  f"{med['rate']:>10.1f}%  "
                  f"{assessment}")
        
        print("\n" + "-" * 80)
        avg_bleu = sum(s['bleu'] for s in stats.values()) / len(stats)
        avg_chrf = sum(s['chrf'] for s in stats.values()) / len(stats)
        avg_bert = sum(s['bertscore_f1'] for s in stats.values()) / len(stats)
        avg_med = sum(m['rate'] for m in med_stats.values()) / len(med_stats) if med_stats else 0
        
        print(f"{'AVERAGE':<15} {avg_bleu:>6.1f}% {avg_chrf:>6.1f}  {avg_bert:>8.1f}  {avg_med:>10.1f}%")
        
        print("\n" + "="*80)
        print("📝 SAMPLE TRANSLATIONS")
        print("="*80)
        
        for lang in sorted(stats.keys())[:2]:
            samples = self.results['meditron_direct'][lang][:2]
            lang_name = HIGH_RESOURCE_LANGS[lang][1]
            
            for i, sample in enumerate(samples, 1):
                print(f"\n{lang_name.upper()} - Sample {i}:")
                print(f"   Source:    {sample['source']}...")
                print(f"   NLLB:      {sample['nllb_reference'][:100]}...")
                print(f"   Meditron:  {sample['prediction'][:100]}...")
        
        print("\n" + "="*80)
        print("💡 CONCLUSION")
        print("="*80)
        
        print(f"\n📊 Quality Assessment:")
        print(f"   Excellent/Good: {winners['meditron']} languages")
        print(f"   Acceptable:     {winners['tie']} languages")
        print(f"   Poor:           {winners['nllb']} languages")
        
        print(f"\n📈 Average Scores:")
        print(f"   BLEU:           {avg_bleu:.1f}%")
        print(f"   chrF:           {avg_chrf:.1f}")
        print(f"   BERTScore F1:   {avg_bert:.1f}")
        print(f"   Med Term Preservation: {avg_med:.1f}%")
        
        print(f"\n🎯 INTERPRETATION:")
        if avg_bert >= 90:
            print(f"   ✅ Meditron's translation is EXCELLENT (avg BERTScore: {avg_bert:.1f})")
            print(f"   → Semantic meaning highly preserved")
            print(f"   → No translation pipeline needed for these languages")
        elif avg_bert >= 80:
            print(f"   ✅ Meditron's translation is GOOD (avg BERTScore: {avg_bert:.1f})")
            print(f"   → Semantic meaning well preserved")
            print(f"   → Translation pipeline provides minimal benefit")
        elif avg_bert >= 70:
            print(f"   ⚠️  Meditron's translation is ACCEPTABLE (avg BERTScore: {avg_bert:.1f})")
            print(f"   → Some semantic drift")
            print(f"   → Consider translation pipeline for critical applications")
        else:
            print(f"   ❌ Meditron's translation is POOR (avg BERTScore: {avg_bert:.1f})")
            print(f"   → Significant semantic loss")
            print(f"   → Translation pipeline REQUIRED")
        
        if det['total'] > 0 and det_acc < 90:
            print(f"\n   ⚠️  WARNING: FastText detection is {det_acc:.1f}%")
            print(f"   → May affect NLLB pipeline reliability")
        
        print("\n" + "="*80)
    
    def save_results(self, output_path: str):
        """Save results to JSON."""
        output = {
            'experiment': 'High-Resource Languages: Meditron vs NLLB',
            'note': 'NLLB used as pseudo-reference (not ground truth)',
            'detection_stats': {
                'overall_accuracy': self.detection_stats['correct'] / self.detection_stats['total'] * 100 if self.detection_stats['total'] > 0 else 0,
                'by_language': {
                    k: {
                        'accuracy': v['correct'] / v['total'] * 100 if v['total'] > 0 else 0,
                        'correct': v['correct'],
                        'total': v['total'],
                        'misdetections': dict(v['misdetections'])
                    }
                    for k, v in self.detection_stats['by_language'].items()
                }
            },
            'translation_stats': self.compute_metrics(),
            'medical_term_preservation': self.analyze_medical_terms(),
            'detailed_results': {
                k: dict(v) for k, v in self.results.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")


def extract_translation(response: str) -> str:
    """Extract English translation from Meditron response."""
    match = re.search(r'translation\s*:\s*[""](.+?)[""]', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'[""](.+?)[""]', response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def translate_with_meditron(model, tokenizer, collator, modality_retriever, 
                           text: str) -> str:
    """Use Meditron to translate text to English."""
    if len(text) > 400:
        text = text[:400]
    
    prompt = f"""You are a professional medical translator. 
Translate the following text into English and return **only** one line in the format:

translation: "your English translation here"

Do not explain or add any other text.

Text to translate:
{text}
"""
    
    conversations = [{"role": "user", "content": prompt}]
    sample = {"conversations": conversations, "modalities": []}
    sample = modality_retriever.merge_modality_with_sample(sample)
    batch = collator([sample])
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            batch=batch,
            temperature=0.1,
            do_sample=False,
            max_new_tokens=150,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = extract_translation(response)
    
    del outputs
    del batch
    del sample
    del conversations
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    
    return translation.strip()


def run_experiment(args):
    """Run high-resource language translation experiment in two passes."""
    print("\n" + "="*80)
    print("EXPERIMENT: HIGH-RESOURCE LANGUAGE TRANSLATION")
    print("Meditron Native vs NLLB Baseline on Medical Q&A")
    print("="*80)
    
    print("\n[1/6] Loading MediBench dataset...")
    dataset = load_medibench_robust(split="train", token=os.getenv("HF_LAB_TOKEN"))
    
    available_langs = set(dataset['language'])
    test_langs = {k: v for k, v in HIGH_RESOURCE_LANGS.items() if k in available_langs}
    
    print(f"   ✅ Loaded {len(dataset)} samples")
    print(f"   📋 Available languages: {list(test_langs.keys())}")
    
    max_samples = args.max_samples
    print(f"   Using {max_samples} samples per language")
    
    print("\n[2/6] Loading fastText...")
    lid_model_path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin"
    )
    fasttext_model = fasttext.load_model(lid_model_path)
    print("   ✅ FastText loaded")
    
    evaluator = HighResourceEvaluator()
    
    print("\n" + "="*80)
    print("[PASS 1/2] MEDITRON TRANSLATION + SAMPLE COLLECTION")
    print("="*80)
    print("\n[3/6] Loading Meditron...")
    
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
        llm_token=os.getenv("HF_PERSONAL_TOKEN"),
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except:
            pass
    
    modality_retriever = ModalityRetriever(
        registry=FileSystemImageRegistry(base_path=os.getcwd())
    )
    collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        tokenizer_type="llama",
        modality_processors=model.processors(),
        attachment_token_idx=attachment_token_idx,
        add_generation_prompt=True
    )
    
    print("   ✅ Meditron loaded")
    
    collected_samples = []
    
    print(f"\n[4/6] Running Meditron translations...")
    for lang_code, (nllb_code, lang_name) in test_langs.items():
        print(f"\n   Processing {lang_name} ({lang_code})...")
        
        lang_samples = [s for s in dataset if s['language'] == lang_code]
        print(f"   Found {len(lang_samples)} samples")
        
        processed = 0
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 10
        
        for idx, sample in enumerate(tqdm(lang_samples, desc=f"  {lang_name}")):
            if processed >= max_samples:
                break
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n      ⚠️  Too many errors, stopping")
                break
            
            if idx % 5 == 0 and idx > 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            question = sample.get('question', '')
            
            if not question or len(question.strip()) < 20:
                continue
            
            if len(question) > 400:
                continue
            
            try:
                pred = fasttext_model.predict(question.replace('\n', ' '), k=1)
                detected_lang = pred[0][0].replace('__label__', '')
                evaluator.track_detection(nllb_code, detected_lang)
                
                meditron_translation = translate_with_meditron(
                    model, tokenizer, collator, modality_retriever,
                    question
                )
                
                collected_samples.append({
                    'lang_code': lang_code,
                    'nllb_code': nllb_code,
                    'source': question,
                    'meditron_translation': meditron_translation,
                    'detected_lang': detected_lang
                })
                
                processed += 1
                consecutive_errors = 0
                
                if processed % 10 == 0:
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    print(f"\n      💾 {processed} samples: {mem_gb:.2f} GB")
                
            except Exception as e:
                consecutive_errors += 1
                print(f"\n      [ERROR] {str(e)[:80]}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        print(f"   ✅ Completed {lang_name}: {processed} samples")
    
    with open("meditron_collected.json", 'w') as f:
        json.dump(collected_samples, f, indent=2, ensure_ascii=False)
    
    print("\n   🗑️  Unloading Meditron...")
    del model, tokenizer, collator, modality_retriever
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n" + "="*80)
    print("[PASS 2/2] NLLB TRANSLATION (Baseline/Pseudo-Reference)")
    print("="*80)
    print("\n[5/6] Loading NLLB...")
    
    # ✅ VERIFIED: Using base NLLB-200 3.3B model
    translator = NLLBTranslator(model_name="facebook/nllb-200-3.3B")
    print("   ✅ NLLB loaded")
    
    print(f"\n   Translating {len(collected_samples)} samples with NLLB...")
    
    for idx, sample in enumerate(tqdm(collected_samples, desc="  NLLB")):
        try:
            nllb_translation = translator.translate(
                sample['source'],
                src_lang=sample['nllb_code'],
                tgt_lang='eng_Latn'
            )
            
            evaluator.add_result(
                pipeline='meditron_direct',
                language=sample['lang_code'],
                source=sample['source'],
                prediction=sample['meditron_translation'],
                nllb_reference=nllb_translation,
                detected_lang=sample['detected_lang']
            )
            
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n      [ERROR] Sample {idx}: {e}")
            torch.cuda.empty_cache()
            continue
    
    print("\n[6/6] Computing results...")
    evaluator.print_summary()
    evaluator.save_results(args.output)
    
    if os.path.exists("meditron_collected.json"):
        os.remove("meditron_collected.json")
    
    print("\n✅ Experiment complete!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Meditron translation on high-resource languages"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Samples per language (default: 1000)"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/base_nllb/experiment_2_results.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    run_experiment(args)