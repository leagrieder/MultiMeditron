"""
NLLB-200 Translator with Language Detection

This module provides a translation interface using the NLLB-200 multilingual model
combined with fastText language detection.

Translation Strategy:
- High confidence detection (>80%): Translate question to English, process, then translate back
- Low confidence detection (≤80%): Pass through as-is to avoid mistranslation of ambiguous text

The confidence threshold helps prevent incorrect translations of short or ambiguous text
while enabling accurate translation of clearly detected languages.

Usage:
    translator = NLLBTranslator()
    english_text = translator.translate_to_english(user_question)
    response_in_user_lang = translator.translate_from_english(english_response)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download


class NLLBTranslator:
    """NLLB-200 translator with fastText language detection."""
    
    def __init__(self, 
                model_name="src/multimeditron/translation/models/nllb-consensus-finetuned-1epoch",  #Fine tuned model - to use the base NLLB-200 3.3B model, add HF path here (nllb-200-3.3B)
                lang_detect_model="facebook/fasttext-language-identification"):
        """Initialize NLLB translator with fastText language detection."""
        print(f"[INFO] Loading NLLB model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        print(f"[INFO] Loading fastText language detection model")
        try:
            import fasttext
            model_path = hf_hub_download(
                repo_id="facebook/fasttext-language-identification",
                filename="model.bin"
            )
            fasttext.FastText.eprint = lambda x: None
            self.lang_detector = fasttext.load_model(model_path)
            print(f"[INFO] fastText model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load fastText: {e}")
            print("[INFO] Ensure: pip install 'numpy<2.0' fasttext")
            raise
        
        self.detected_user_lang = None
        print(f"[INFO] NLLB translator ready on {self.device}")
    
    def detect_language(self, text: str, confidence_threshold=0.80) -> str:
        """
        Detect language using fastText. Returns 'eng_Latn' if confidence < threshold
        to trigger pass-through behavior (no translation).
        """
        try:
            clean_text = text.replace('\n', ' ').strip()
            predictions = self.lang_detector.predict(clean_text, k=3)
            
            detected_code = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])
            
            print(f"[DEBUG] Detected: {detected_code} (confidence: {confidence:.3f})")
            
            if confidence < confidence_threshold:
                print(f"[WARNING] Low confidence ({confidence:.3f} < {confidence_threshold}).")
                print(f"[INFO] Defaulting to eng_Latn - text will pass through as-is.")
                print(f"  Top predictions:")
                for i in range(min(3, len(predictions[0]))):
                    alt_code = predictions[0][i].replace('__label__', '')
                    alt_conf = float(predictions[1][i])
                    print(f"    {i+1}. {alt_code}: {alt_conf:.3f}")
                return 'eng_Latn'
            
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(detected_code)
                if token_id == self.tokenizer.unk_token_id:
                    print(f"[WARNING] '{detected_code}' not supported. Defaulting to eng_Latn.")
                    return 'eng_Latn'
            except Exception as e:
                print(f"[WARNING] Validation failed for '{detected_code}'. Defaulting to eng_Latn.")
                return 'eng_Latn'
            
            return detected_code
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}. Defaulting to eng_Latn.")
            return 'eng_Latn'
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from src_lang to tgt_lang using NLLB."""
        if not text or not text.strip():
            return text
        
        if src_lang == tgt_lang:
            return text
        
        self.tokenizer.src_lang = src_lang
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        try:
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        except Exception as e:
            print(f"[ERROR] Failed to get token ID for {tgt_lang}: {e}")
            return text
        
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        result = decoded[0]
        
        text_preview = text[:80] + '...' if len(text) > 80 else text
        result_preview = result[:80] + '...' if len(result) > 80 else result
        print(f"  Input:  {text_preview}")
        print(f"  Output: {result_preview}")
        
        return result
    
    def translate_to_english(self, text: str, src_lang: str = None) -> str:
        """
        Translate to English if high confidence detection, otherwise pass through.
        Stores detected language for translate_from_english().
        """
        if src_lang is None:
            src_lang = self.detect_language(text)
        
        self.detected_user_lang = src_lang
        
        if src_lang == 'eng_Latn':
            return text
        
        return self.translate(text, src_lang, 'eng_Latn')
    
    def translate_from_english(self, text: str, tgt_lang: str = None) -> str:
        """
        Translate from English back to original language.
        If original was low confidence (eng_Latn), passes through unchanged.
        """
        if tgt_lang is None:
            if self.detected_user_lang is None:
                print("[WARNING] No detected language stored. Returning as-is.")
                return text
            tgt_lang = self.detected_user_lang
        
        if tgt_lang == 'eng_Latn':
            return text
        
        return self.translate(text, 'eng_Latn', tgt_lang)