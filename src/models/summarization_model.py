"""Text summarization models using BERT/T5."""

import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
from src.config import HF_TOKEN
from huggingface_hub import login

# Initialize HuggingFace login
login(token=HF_TOKEN)

# Model cache for summarization
SUMMARY_MODELS = {'vi': None, 'en': None}
SUMMARY_TOKENIZERS = {'vi': None, 'en': None}
model_lock = threading.Lock()

def get_summarization_model(lang):
    """Get or initialize the summarization model for a specific language."""
    model_name = "vinai/bartpho-syllable" if lang == 'vi' else "t5-base"
    
    with model_lock:
        if SUMMARY_MODELS[lang] is None:
            if lang == 'en':
                # Sử dụng lớp cụ thể cho T5 với model_max_length phù hợp
                tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024)
                model = T5ForConditionalGeneration.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            SUMMARY_MODELS[lang] = model
            SUMMARY_TOKENIZERS[lang] = tokenizer
    
    return SUMMARY_MODELS[lang], SUMMARY_TOKENIZERS[lang]

def summarize_text(text, lang, max_length=150, min_length=30):
    """Summarize text using the appropriate model for the language."""
    try:
        model, tokenizer = get_summarization_model(lang)
        device = torch.device("cpu")
        model = model.to(device)
        
        # Tạo một bản sao của tokenizer để tránh lỗi "Already borrowed"
        if lang == 'en':
            # Tạo tokenizer mới cho mỗi lần gọi để tránh lỗi "Already borrowed"
            tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=1024)
        
        # Process text in chunks to handle long texts
        max_tokens = 512 if lang == 'en' else 768
        
        # Chia văn bản thành các đoạn nhỏ hơn
        # Thay vì chia theo câu, chia theo số lượng ký tự để đơn giản hóa
        text_length = len(text)
        chunk_size = max_tokens * 4  # Ước tính khoảng 4 ký tự cho mỗi token
        
        chunks = []
        for i in range(0, text_length, chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        
        if not chunks:
            chunks = [text]
        
        # Generate summary for each chunk
        summaries = []
        for chunk in chunks:
            try:
                # Tạo input mới cho mỗi chunk
                if lang == 'en':
                    # For T5 model
                    prefix_text = "summarize: " + chunk
                    inputs = tokenizer(prefix_text, return_tensors="pt", max_length=max_tokens, truncation=True)
                else:
                    # For BARTpho model
                    inputs = tokenizer(chunk, return_tensors="pt", max_length=max_tokens, truncation=True)
                
                inputs = inputs.to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=max_length,
                        min_length=min_length,
                        num_beams=4,
                        early_stopping=True
                    )
                
                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                summaries.append(summary)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        if not summaries:
            return "Could not generate summary due to an error."
        
        return " ".join(summaries)
    except Exception as e:
        import traceback
        print(f"Error in summarize_text: {str(e)}")
        traceback.print_exc()
        # Trả về một tóm tắt đơn giản nếu có lỗi
        return "Could not generate summary due to an error."




