#!/usr/bin/env python3
"""
Multi-model translation pipeline with BATCHED processing.
Uses Qwen2.5, M2M100, and MADLAD-400 with consensus selection.
"""

import json
import os
import gc
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import argparse

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    AutoModelForCausalLM
)
import torch
from sacrebleu.metrics import BLEU, CHRF

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed, using fallback English detection")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def is_english(text: str, threshold: float = 0.7) -> bool:
    """Check if text is primarily English."""
    if not text or len(text.strip()) < 10:
        return True
    
    text = text.strip()
    
    if LANGDETECT_AVAILABLE:
        try:
            return detect(text) == 'en'
        except LangDetectException:
            pass
    
    common_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
        'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
        'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over'
    }
    
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return False
    
    english_count = sum(1 for w in words if w in common_words)
    return (english_count / len(words)) > threshold


def detect_repetition(text: str, word_threshold: float = 0.4, ngram_threshold: int = 4) -> bool:
    """Check for excessive repetition."""
    if not text or len(text.strip()) < 20:
        return False
    
    words = text.strip().split()
    if len(words) < 5:
        return False
    
    word_counts = {}
    for word in words:
        word_lower = word.lower()
        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
    
    if max(word_counts.values()) / len(words) > word_threshold:
        return True
    
    for n in [2, 3, 4]:
        if len(words) < n:
            continue
        
        ngrams = {}
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        
        if ngrams and max(ngrams.values()) >= ngram_threshold:
            return True
    
    if re.search(r'(.)\1{5,}', text):
        return True
    
    return False


def clean_llm_wrapper_text(text: str) -> str:
    """Remove LLM preambles and formatting."""
    if not text:
        return text
    
    text = text.strip()
    
    patterns = [
        r'^here\s+is\s+the\s+translation:?\s*',
        r'^here\'s\s+the\s+translation:?\s*',
        r'^translation:?\s*',
        r'^sure,?\s+here\s+is\s+',
        r'^sure,?\s+here\'s\s+',
        r'^certainly,?\s*',
        r'^of\s+course,?\s*',
        r'^yes,?\s+i\s+can\s+translate\s+',
        r'^yes,?\s+',
        r'^the\s+translation\s+is:?\s*',
        r'^okay,?\s*',
        r'^alright,?\s*',
        r'^i\s+will\s+translate\s+',
        r'^i\'ll\s+translate\s+',
        r'^translating\s+to\s+\w+:?\s*',
        r'^in\s+\w+:?\s*',
        r'^translated:?\s*',
        r'^output:?\s*',
        r'^result:?\s*',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = text.strip()
    
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    
    text = re.sub(r'^\*\*(.+)\*\*$', r'\1', text)
    text = re.sub(r'^`(.+)`$', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_translation(text: str) -> Tuple[bool, str]:
    """Validate a single translation. Returns (is_valid, status)."""
    if not text or len(text) < 3:
        return False, "error"
    
    if is_english(text):
        return False, "english"
    
    if detect_repetition(text):
        return False, "repetitive"
    
    return True, "success"


def translate_batch_qwen(
    texts: List[str],
    target_lang: str,
    model,
    tokenizer,
    lang_name: str,
    batch_size: int = 32,
    max_attempts: int = 5
) -> Tuple[List[str], List[str]]:
    """Translate a batch of texts with Qwen, with retry logic."""
    
    translations = [""] * len(texts)
    statuses = [""] * len(texts)
    retry_indices = list(range(len(texts)))
    
    system_msg = (
        f"You are a professional translation system. "
        f"Translate English medical text to {lang_name}. "
        f"Rules:\n"
        f"1. Output ONLY the {lang_name} translation\n"
        f"2. Do NOT add any explanations, acknowledgments, or English text\n"
        f"3. Do NOT say 'here is the translation' or similar phrases\n"
        f"4. Do NOT wrap the translation in quotes\n"
        f"5. Preserve all medical terminology accurately"
    )
    
    for attempt in range(1, max_attempts + 1):
        if not retry_indices:
            break
        
        batch_texts = [texts[i] for i in retry_indices]
        batch_results = []
        
        # Process in mini-batches
        for i in range(0, len(batch_texts), batch_size):
            mini_batch = batch_texts[i:i+batch_size]
            
            try:
                # Create prompts for all texts in mini-batch
                prompts = []
                for text in mini_batch:
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"Translate this to {lang_name}:\n\n{text}"}
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                
                # Tokenize batch
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate batch
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decode batch
                for j, output in enumerate(outputs):
                    input_len = inputs['input_ids'][j].shape[0]
                    generated_ids = output[input_len:]
                    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    result = clean_llm_wrapper_text(result)
                    batch_results.append(result)
                
            except Exception as e:
                print(f"    Batch error (attempt {attempt}): {e}")
                batch_results.extend([""] * len(mini_batch))
        
        # Validate results and update retry list
        new_retry_indices = []
        for idx, orig_idx in enumerate(retry_indices):
            result = batch_results[idx]
            is_valid, status = validate_translation(result)
            
            if is_valid or attempt == max_attempts:
                translations[orig_idx] = result
                statuses[orig_idx] = status
            else:
                new_retry_indices.append(orig_idx)
        
        retry_indices = new_retry_indices
    
    # Mark any remaining failures
    for idx in retry_indices:
        if not statuses[idx]:
            statuses[idx] = "error"
    
    return translations, statuses


def translate_with_qwen(
    texts: List[str],
    target_lang: str,
    model_size: str = "3B-Instruct",
    batch_size: int = 32
) -> Tuple[List[str], List[str]]:
    """Batch translate with Qwen."""
    print(f"\n[1/3] Loading Qwen2.5-{model_size}...")
    
    lang_names = {
        "amh": "Amharic", "swa": "Swahili", "tam": "Tamil",
        "hau": "Hausa", "yor": "Yoruba", "zul": "Zulu"
    }
    lang_name = lang_names.get(target_lang, target_lang)
    
    tokenizer = AutoTokenizer.from_pretrained(f"Qwen/Qwen2.5-{model_size}", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        f"Qwen/Qwen2.5-{model_size}",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Translating {len(texts)} texts (batch_size={batch_size})...")
    translations, statuses = translate_batch_qwen(texts, target_lang, model, tokenizer, lang_name, batch_size)
    
    status_counts = {"success": 0, "english": 0, "repetitive": 0, "error": 0}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    
    del model, tokenizer
    clear_gpu_memory()
    
    print(f"Qwen2.5 complete: {status_counts['success']} ok, {status_counts['english']} english, "
          f"{status_counts['repetitive']} repetitive, {status_counts['error']} errors")
    
    return translations, statuses


def translate_batch_m2m100(
    texts: List[str],
    target_lang: str,
    model,
    tokenizer,
    tgt_lang: str,
    batch_size: int = 32,
    max_attempts: int = 5
) -> Tuple[List[str], List[str]]:
    """Translate batch with M2M100, with retry logic."""
    
    translations = [""] * len(texts)
    statuses = [""] * len(texts)
    retry_indices = list(range(len(texts)))
    
    tokenizer.src_lang = "en"
    
    for attempt in range(1, max_attempts + 1):
        if not retry_indices:
            break
        
        batch_texts = [texts[i] for i in retry_indices]
        batch_results = []
        
        for i in range(0, len(batch_texts), batch_size):
            mini_batch = batch_texts[i:i+batch_size]
            
            try:
                encoded = tokenizer(
                    mini_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                    max_length=512,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3
                )
                
                results = tokenizer.batch_decode(generated, skip_special_tokens=True)
                batch_results.extend([r.strip() for r in results])
                
            except Exception as e:
                print(f"    Batch error (attempt {attempt}): {e}")
                batch_results.extend([""] * len(mini_batch))
        
        new_retry_indices = []
        for idx, orig_idx in enumerate(retry_indices):
            result = batch_results[idx]
            is_valid, status = validate_translation(result)
            
            if is_valid or attempt == max_attempts:
                translations[orig_idx] = result
                statuses[orig_idx] = status
            else:
                new_retry_indices.append(orig_idx)
        
        retry_indices = new_retry_indices
    
    for idx in retry_indices:
        if not statuses[idx]:
            statuses[idx] = "error"
    
    return translations, statuses


def translate_with_m2m100(
    texts: List[str],
    target_lang: str,
    batch_size: int = 32
) -> Tuple[List[str], List[str]]:
    """Batch translate with M2M100."""
    print(f"\n[2/3] Loading M2M100...")
    
    lang_codes = {"amh": "am", "swa": "sw", "tam": "ta", "hau": "ha", "yor": "yo", "zul": "zu"}
    tgt_lang = lang_codes.get(target_lang, target_lang)
    
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained(
        "facebook/m2m100_418M",
        torch_dtype=torch.float16
    ).to(device)
    
    print(f"Translating {len(texts)} texts (batch_size={batch_size})...")
    translations, statuses = translate_batch_m2m100(texts, target_lang, model, tokenizer, tgt_lang, batch_size)
    
    status_counts = {"success": 0, "english": 0, "repetitive": 0, "error": 0}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    
    del model, tokenizer
    clear_gpu_memory()
    
    print(f"M2M100 complete: {status_counts['success']} ok, {status_counts['english']} english, "
          f"{status_counts['repetitive']} repetitive, {status_counts['error']} errors")
    
    return translations, statuses


def translate_batch_madlad(
    texts: List[str],
    target_lang: str,
    model,
    tokenizer,
    tgt_lang: str,
    batch_size: int = 32,
    max_attempts: int = 5
) -> Tuple[List[str], List[str]]:
    """Translate batch with MADLAD, with retry logic."""
    
    translations = [""] * len(texts)
    statuses = [""] * len(texts)
    retry_indices = list(range(len(texts)))
    
    for attempt in range(1, max_attempts + 1):
        if not retry_indices:
            break
        
        batch_texts = [texts[i] for i in retry_indices]
        batch_results = []
        
        for i in range(0, len(batch_texts), batch_size):
            mini_batch = batch_texts[i:i+batch_size]
            
            try:
                prompts = [f"<2{tgt_lang}> {text}" for text in mini_batch]
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3
                )
                
                results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_results.extend([r.strip() for r in results])
                
            except Exception as e:
                print(f"    Batch error (attempt {attempt}): {e}")
                batch_results.extend([""] * len(mini_batch))
        
        new_retry_indices = []
        for idx, orig_idx in enumerate(retry_indices):
            result = batch_results[idx]
            is_valid, status = validate_translation(result)
            
            if is_valid or attempt == max_attempts:
                translations[orig_idx] = result
                statuses[orig_idx] = status
            else:
                new_retry_indices.append(orig_idx)
        
        retry_indices = new_retry_indices
    
    for idx in retry_indices:
        if not statuses[idx]:
            statuses[idx] = "error"
    
    return translations, statuses


def translate_with_madlad(
    texts: List[str],
    target_lang: str,
    batch_size: int = 32
) -> Tuple[List[str], List[str]]:
    """Batch translate with MADLAD."""
    print(f"\n[3/3] Loading MADLAD-400...")
    
    lang_codes = {"amh": "amh", "swa": "swa", "tam": "tam", "hau": "hau", "yor": "yor", "zul": "zul"}
    tgt_lang = lang_codes.get(target_lang, target_lang)
    
    tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/madlad400-3b-mt",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Translating {len(texts)} texts (batch_size={batch_size})...")
    translations, statuses = translate_batch_madlad(texts, target_lang, model, tokenizer, tgt_lang, batch_size)
    
    status_counts = {"success": 0, "english": 0, "repetitive": 0, "error": 0}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1
    
    del model, tokenizer
    clear_gpu_memory()
    
    print(f"MADLAD-400 complete: {status_counts['success']} ok, {status_counts['english']} english, "
          f"{status_counts['repetitive']} repetitive, {status_counts['error']} errors")
    
    return translations, statuses


def compute_consensus_with_fallback(
    all_translations: Dict[str, List[str]],
    all_statuses: Dict[str, List[str]]
) -> Tuple[List[str], List[str], List[Dict], List[str], List[int]]:
    """Select best translation per entry using BLEU/chrF consensus."""
    bleu = BLEU(effective_order=True)
    chrf = CHRF()
    
    num_entries = len(all_translations["qwen2.5"])
    best_translations = []
    best_models = []
    all_scores = []
    final_statuses = []
    skipped_indices = []
    
    for i in range(num_entries):
        translations = {
            "qwen2.5": all_translations["qwen2.5"][i],
            "m2m100": all_translations["m2m100"][i],
            "madlad-400": all_translations["madlad-400"][i]
        }
        
        statuses = {
            "qwen2.5": all_statuses["qwen2.5"][i],
            "m2m100": all_statuses["m2m100"][i],
            "madlad-400": all_statuses["madlad-400"][i]
        }
        
        successful = {k: v for k, v in translations.items() if v and statuses[k] == "success"}
        
        if successful:
            if len(successful) == 1:
                model = list(successful.keys())[0]
                best_translations.append(successful[model])
                best_models.append(model)
                all_scores.append({model: 100.0})
                final_statuses.append("success")
                continue
            
            scores = {}
            for model, candidate in successful.items():
                refs = [t for m, t in successful.items() if m != model]
                try:
                    b = bleu.sentence_score(candidate, refs).score
                    c = chrf.sentence_score(candidate, refs).score
                    scores[model] = 0.5 * b + 0.5 * c
                except:
                    scores[model] = 0.0
            
            best_model = max(scores, key=scores.get)
            best_translations.append(successful[best_model])
            best_models.append(best_model)
            all_scores.append(scores)
            final_statuses.append("success")
            continue
        
        # All models failed - skip
        skipped_indices.append(i)
        best_translations.append(None)
        best_models.append("skipped")
        all_scores.append({})
        final_statuses.append("skipped")
    
    return best_translations, best_models, all_scores, final_statuses, skipped_indices


def main():
    parser = argparse.ArgumentParser(description="Multi-model translation pipeline with batching")
    parser.add_argument("--input", type=str,
                       default="/mloscratch/users/nemo/datasets/guidelines/chunked_eng.jsonl")
    parser.add_argument("--output", type=str,
                       default="src/multimeditron/translation/datasets/generated_datasets/consensus")
    parser.add_argument("--languages", nargs="+", default=["amh", "swa", "tam", "hau", "yor", "zul"])
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--qwen-size", type=str, default="3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for translation")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Multi-Model Translation Pipeline (BATCHED)")
    print("Models: Qwen2.5, M2M100, MADLAD-400")
    print(f"Batch size: {args.batch_size}")
    print("="*60)
    
    print(f"\nLoading dataset from {args.input}...")
    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if args.sample:
        data = data[:args.sample]
    
    texts = [entry.get('text', entry.get('content', '')) for entry in data]
    texts = [t for t in texts if t]
    print(f"Loaded {len(texts)} texts")
    
    for lang in args.languages:
        print(f"\n{'='*60}")
        print(f"Processing {lang.upper()}")
        print(f"{'='*60}")
        
        all_translations = {}
        all_statuses = {}
        
        qwen_trans, qwen_status = translate_with_qwen(texts, lang, args.qwen_size, args.batch_size)
        all_translations["qwen2.5"] = qwen_trans
        all_statuses["qwen2.5"] = qwen_status
        
        m2m_trans, m2m_status = translate_with_m2m100(texts, lang, args.batch_size)
        all_translations["m2m100"] = m2m_trans
        all_statuses["m2m100"] = m2m_status
        
        madlad_trans, madlad_status = translate_with_madlad(texts, lang, args.batch_size)
        all_translations["madlad-400"] = madlad_trans
        all_statuses["madlad-400"] = madlad_status
        
        print(f"\nComputing consensus...")
        best_translations, best_models, scores, final_statuses, skipped = compute_consensus_with_fallback(
            all_translations, all_statuses
        )
        
        output_file = Path(args.output) / f"{lang}_train.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        with open(output_file, 'w') as f:
            for i, entry in enumerate(data[:len(texts)]):
                if i in skipped:
                    continue
                
                output = {
                    **entry,
                    "translated_text": best_translations[i],
                    "target_language": lang,
                    "source_language": "eng",
                    "translation_model": best_models[i],
                    "translation_status": final_statuses[i],
                    "consensus_scores": scores[i],
                    "all_translations": {
                        "qwen2.5": all_translations["qwen2.5"][i],
                        "m2m100": all_translations["m2m100"][i],
                        "madlad-400": all_translations["madlad-400"][i]
                    },
                    "all_statuses": {
                        "qwen2.5": all_statuses["qwen2.5"][i],
                        "m2m100": all_statuses["m2m100"][i],
                        "madlad-400": all_statuses["madlad-400"][i]
                    }
                }
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
                saved_count += 1
        
        print(f"\nSaved: {output_file}")
        print(f"Saved {saved_count} translations (skipped {len(skipped)} failed entries)")
        
        valid_models = [m for i, m in enumerate(best_models) if i not in skipped]
        valid_statuses = [s for i, s in enumerate(final_statuses) if i not in skipped]
        
        model_counts = {}
        for m in valid_models:
            model_counts[m] = model_counts.get(m, 0) + 1
        
        print(f"\nModel distribution:")
        for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
            pct = count / len(valid_models) * 100 if valid_models else 0
            print(f"  {model:15s}: {count:5d} ({pct:5.1f}%)")
        
        status_counts = {}
        for s in valid_statuses:
            status_counts[s] = status_counts.get(s, 0) + 1
        
        print(f"\nFinal status:")
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            pct = count / len(valid_statuses) * 100 if valid_statuses else 0
            print(f"  {status:20s}: {count:5d} ({pct:5.1f}%)")
        
        successful_scores = [s[m] for i, (s, m, fs) in enumerate(zip(scores, best_models, final_statuses))
                           if i not in skipped and m in s and fs == "success"]
        if successful_scores:
            print(f"\nAverage consensus score: {np.mean(successful_scores):.2f}")
    
    print("\n" + "="*60)
    print("Complete")
    print("="*60)


if __name__ == "__main__":
    main()