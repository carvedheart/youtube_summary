"""Text cleaning and processing utilities."""

import re
from langdetect import detect, LangDetectException

def clean_subtitle(text):
    """Clean subtitle text by removing timestamps and formatting."""
    if not text:
        return ""
        
    # Remove subtitle numbers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove timestamps
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def clean_whisper_output(text):
    """Clean Whisper transcription output by removing duplicates and noise."""
    if not text:
        return ""
        
    # Split into sentences
    sentences = text.split('. ')
    
    # Remove duplicate sentences
    unique_sentences = []
    seen = set()
    
    for sent in sentences:
        # Create a simplified version for comparison (lowercase, no punctuation)
        simplified = re.sub(r'\W+', '', sent.lower())
        
        if simplified not in seen and len(sent.split()) > 3:
            seen.add(simplified)
            unique_sentences.append(sent)
    
    # Remove repeated words within sentences
    cleaned_text = []
    for sent in unique_sentences:
        words = sent.split()
        unique_words = []
        prev_word = None
        
        for word in words:
            if word != prev_word:
                unique_words.append(word)
            prev_word = word
            
        cleaned_text.append(" ".join(unique_words))
        
    return ". ".join(cleaned_text)

def detect_language(text):
    """Detect language of text, focusing on Vietnamese and English."""
    if not text or len(text.strip()) < 10:
        return None
        
    try:
        lang = detect(text)
        return lang if lang in ['vi', 'en'] else None
    except LangDetectException:
        return None

def truncate_text(text, max_length=1000):
    """Truncate text to maximum length while preserving complete sentences."""
    if not text or len(text) <= max_length:
        return text
        
    # Find the last period within the max_length
    last_period = text[:max_length].rfind('.')
    
    if last_period > 0:
        return text[:last_period + 1]
    else:
        # If no period found, just truncate at max_length
        return text[:max_length]