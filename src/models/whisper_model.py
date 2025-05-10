"""Whisper model handling for transcription."""

from functools import lru_cache
from faster_whisper import WhisperModel
from src.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

@lru_cache(maxsize=1)
def get_whisper_model():
    """Get or initialize the Whisper model with caching."""
    print(f"Initializing Whisper model: size={WHISPER_MODEL_SIZE}, device={WHISPER_DEVICE}, compute_type={WHISPER_COMPUTE_TYPE}")
    return WhisperModel(
        WHISPER_MODEL_SIZE, 
        device=WHISPER_DEVICE, 
        compute_type=WHISPER_COMPUTE_TYPE
    )

def transcribe_audio(audio_path, language=None):
    """Transcribe audio file using Whisper model."""
    try:
        model = get_whisper_model()
        
        # Kiểm tra file audio
        import os
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return []
            
        print(f"Audio file size: {os.path.getsize(audio_path)} bytes")
        
        # Sửa lỗi: Nếu language là 'auto', đặt thành None để Whisper tự động nhận diện
        if language == 'auto':
            language = None
            
        print(f"Starting transcription with language={language}")
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            temperature=0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            initial_prompt=None
        )
        
        print(f"Transcription info: {info}")
        segments_list = list(segments)
        print(f"Transcribed {len(segments_list)} segments")
        return segments_list
    except Exception as e:
        print(f"Whisper transcription error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

