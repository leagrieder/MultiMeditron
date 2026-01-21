"""
NLLB-200-3.3B Fine-tuning - Multi-GPU Optimized
Using consensus translations from Qwen2.5, M2M100, MADLAD-400
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

print("="*70)
print("NLLB Fine-tuning - Multi-GPU Optimized")
print("Languages: Amharic, Swahili, Tamil, Hausa, Yoruba, Zulu")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# =============================================================================
# CONFIGURATION - MULTI-GPU OPTIMIZED
# =============================================================================

DATA_DIR = "src/multimeditron/translation/datasets/generated_datasets/consensus"
OUTPUT_DIR = "src/multimeditron/translation/models/nllb-consensus-finetuned"
MODEL_NAME = "facebook/nllb-200-3.3B"

LANGUAGE_MAPPING = {
    "amh": "amh_Ethi",
    "swa": "swh_Latn",
    "tam": "tam_Taml",
    "hau": "hau_Latn",
    "yor": "yor_Latn",
    "zul": "zul_Latn",
}

# Multi-GPU optimized settings
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"\n🖥️  Detected {num_gpus} GPU(s)")

# H100 80GB optimized settings
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4

NUM_EPOCHS = 5
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256

SAMPLES_PER_LANGUAGE = None

SAVE_EVERY = 200
LOG_EVERY = 25
EVAL_STEPS = 100

effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION * num_gpus

print(f"\n📊 Configuration:")
print(f"  GPUs: {num_gpus}")
print(f"  Batch per GPU: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective batch size: {effective_batch}")
print(f"  Languages: {len(LANGUAGE_MAPPING)}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Warmup ratio: {WARMUP_RATIO}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "training_config.json"), 'w') as f:
    json.dump({
        "version": "consensus_multi_gpu",
        "num_gpus": num_gpus,
        "batch_per_gpu": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": effective_batch,
        "languages": list(LANGUAGE_MAPPING.keys()),
        "samples_per_language": SAMPLES_PER_LANGUAGE,
        "start_time": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "data_source": "consensus_translations_qwen_m2m_madlad",
    }, f, indent=2)

# =============================================================================
# LOAD DATASETS
# =============================================================================

print(f"\n📂 Loading consensus translation datasets...")

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

all_data = []
language_counts = {}
skipped_counts = {}

for lang_code, nllb_code in LANGUAGE_MAPPING.items():
    filepath = os.path.join(DATA_DIR, f"{lang_code}_train.jsonl")
    
    if not os.path.exists(filepath):
        print(f"⚠️  Warning: {filepath} not found, skipping {lang_code}")
        continue
    
    print(f"  Loading {lang_code}...")
    lang_data = load_jsonl(filepath)
    
    if SAMPLES_PER_LANGUAGE:
        lang_data = lang_data[:SAMPLES_PER_LANGUAGE]
        print(f"    Sampled {len(lang_data)} entries")
    
    skipped = 0
    for entry in lang_data:
        eng_text = entry.get('text', entry.get('content', ''))
        translation = entry.get('translated_text', '')
        
        if not eng_text or not translation:
            skipped += 1
            continue
        
        status = entry.get('translation_status', '')
        if status != 'success':
            skipped += 1
            continue
        
        all_data.append({
            'english': eng_text,
            'translation': translation,
            'target_lang': nllb_code,
            'source_lang_code': lang_code,
            'translation_model': entry.get('translation_model', ''),
            'consensus_scores': entry.get('consensus_scores', {}),
        })
    
    language_counts[lang_code] = len([d for d in all_data if d['source_lang_code'] == lang_code])
    skipped_counts[lang_code] = skipped
    print(f"    ✓ {language_counts[lang_code]} examples (skipped {skipped} low-quality)")

print(f"\n✓ Total examples loaded: {len(all_data):,}")
print(f"\n📊 Language distribution:")
for lang, count in language_counts.items():
    pct = count / len(all_data) * 100
    print(f"  {lang}: {count:,} ({pct:.1f}%)")

# =============================================================================
# TRAIN/VAL SPLIT
# =============================================================================

print(f"\n✂️  Creating train/validation split...")

dataset = Dataset.from_list(all_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"  Train: {len(train_dataset):,}")
print(f"  Validation: {len(eval_dataset):,}")

# =============================================================================
# TOKENIZER
# =============================================================================

print(f"\n🔤 Loading tokenizer...")
tokenizer = NllbTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang="eng_Latn",
    cache_dir=os.environ.get("HF_HOME", None)
)
print(f"✓ Tokenizer loaded")

# =============================================================================
# PREPROCESSING - BIDIRECTIONAL
# =============================================================================

print(f"\n⚙️  Preprocessing (bidirectional training)...")

def preprocess_function(examples):
    """Creates BOTH EN→X and X→EN examples."""
    sources = []
    targets = []
    src_langs = []
    tgt_langs = []
    
    for eng, trans, tgt_lang in zip(
        examples['english'],
        examples['translation'],
        examples['target_lang']
    ):
        if not eng or not trans:
            continue
        
        # EN → Target
        sources.append(eng)
        targets.append(trans)
        src_langs.append('eng_Latn')
        tgt_langs.append(tgt_lang)
        
        # Target → EN
        sources.append(trans)
        targets.append(eng)
        src_langs.append(tgt_lang)
        tgt_langs.append('eng_Latn')
    
    model_inputs = tokenizer(
        sources,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    labels_list = []
    for target, tgt_lang in zip(targets, tgt_langs):
        tokenizer.src_lang = tgt_lang
        labels = tokenizer(
            target,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
        labels_list.append(labels["input_ids"])
    
    model_inputs["labels"] = labels_list
    return model_inputs

# Use multiple workers for preprocessing on multi-GPU
num_proc = min(num_gpus * 4, 16)  # 4 workers per GPU, max 16

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
    num_proc=num_proc
)

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing validation",
    num_proc=num_proc
)

print(f"✓ Train: {len(tokenized_train):,} examples (bidirectional)")
print(f"✓ Validation: {len(tokenized_eval):,} examples (bidirectional)")

# =============================================================================
# MODEL
# =============================================================================

print(f"\n🤖 Loading model: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir=os.environ.get("HF_HOME", None)
)

if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    print(f"  ✓ Gradient checkpointing enabled")

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model loaded: {num_params/1e9:.2f}B parameters")

# =============================================================================
# METRICS
# =============================================================================

print(f"\n📏 Loading evaluation metrics...")
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(eval_preds):
    """Compute BLEU and chrF scores"""
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    vocab_size = len(tokenizer)
    preds = np.clip(preds, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)
    
    try:
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"\n⚠️  Decoding error: {e}")
        return {"bleu": 0.0, "chrf": 0.0}
    
    valid_pairs = [
        (pred, ref) for pred, ref in zip(decoded_preds, decoded_labels)
        if pred.strip() and ref.strip()
    ]
    
    if not valid_pairs:
        return {"bleu": 0.0, "chrf": 0.0}
    
    decoded_preds, decoded_labels = zip(*valid_pairs)
    
    try:
        bleu_result = bleu_metric.compute(
            predictions=list(decoded_preds),
            references=[[label] for label in decoded_labels]
        )
        
        chrf_result = chrf_metric.compute(
            predictions=list(decoded_preds),
            references=list(decoded_labels)
        )
        
        return {
            "bleu": bleu_result["bleu"] * 100,
            "chrf": chrf_result["score"],
        }
    except Exception as e:
        print(f"\n⚠️  Metric computation error: {e}")
        return {"bleu": 0.0, "chrf": 0.0}

print(f"✓ Metrics: BLEU & chrF")

# =============================================================================
# TRAINING SETUP
# =============================================================================

print(f"\n🎯 Setting up training...")

steps_per_epoch = len(tokenized_train) // effective_batch
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

print(f"  Effective batch: {effective_batch}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")

# Estimate time based on number of GPUs
time_per_step = 1.2  # H100 is fast
estimated_hours = total_steps * time_per_step / 3600
print(f"  Estimated time: ~{estimated_hours:.1f} hours on {num_gpus} GPU(s)")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training schedule
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    
    # Optimization
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,
    
    # Precision
    fp16=torch.cuda.is_available(),
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    
    # Saving
    save_strategy="steps",
    save_steps=SAVE_EVERY,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_chrf",
    greater_is_better=True,
    
    # Logging
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=LOG_EVERY,
    report_to=["tensorboard"],
    
    # Generation
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    generation_num_beams=4,
    
    # Performance - Multi-GPU optimized
    dataloader_num_workers=min(num_gpus * 4, 16),  # Scale workers with GPUs
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    dataloader_prefetch_factor=2,  # Prefetch batches
    
    # Multi-GPU
    ddp_find_unused_parameters=False,  # Faster multi-GPU training
    
    # Misc
    push_to_hub=False,
    seed=42,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

print(f"✓ Trainer ready")
print(f"✓ Multi-GPU training will use all {num_gpus} GPU(s) automatically")

# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)
print(f"\nDataset: {len(all_data):,} high-quality translations")
print(f"Languages: 6 African/Asian languages")
print(f"Training: Bidirectional (EN↔Target)")
print(f"GPUs: {num_gpus}")
print(f"Effective batch: {effective_batch}")
print(f"\nMonitor: tensorboard --logdir {OUTPUT_DIR}/logs")
print("="*70 + "\n")

try:
    train_result = trainer.train()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    
    print(f"\n💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    metadata = {
        "version": "consensus_multi_gpu",
        "status": "completed",
        "num_gpus": num_gpus,
        "effective_batch_size": effective_batch,
        "languages": list(LANGUAGE_MAPPING.keys()),
        "samples_per_language": SAMPLES_PER_LANGUAGE,
        "language_counts": language_counts,
        "skipped_counts": skipped_counts,
        "end_time": datetime.now().isoformat(),
        "total_steps": train_result.metrics.get('train_steps', 'N/A'),
        "epochs_completed": train_result.metrics.get('epoch', 'N/A'),
        "final_train_loss": train_result.metrics.get('train_loss', 'N/A'),
        "best_eval_bleu": train_result.metrics.get('eval_bleu', 'N/A'),
        "best_eval_chrf": train_result.metrics.get('eval_chrf', 'N/A'),
        "training_hours": train_result.metrics.get('train_runtime', 0) / 3600,
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    
    print(f"\n📊 Training Statistics:")
    print(f"  GPUs used: {num_gpus}")
    print(f"  Total steps: {metadata['total_steps']}")
    print(f"  Epochs: {metadata['epochs_completed']:.2f}")
    print(f"  Final loss: {metadata['final_train_loss']:.4f}")
    print(f"  Best BLEU: {metadata['best_eval_bleu']:.2f}")
    print(f"  Best chrF: {metadata['best_eval_chrf']:.2f}")
    print(f"  Time: {metadata['training_hours']:.2f} hours")
    
    print(f"\n✅ Model saved to: {OUTPUT_DIR}")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted")
    
except Exception as e:
    print(f"\n\n❌ Training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*70)