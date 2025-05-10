"""Video processing logic for YouTube videos."""

import os
import re
from langdetect import detect

from src.models.whisper_model import get_whisper_model, transcribe_audio
from src.models.summarization_model import summarize_text
from src.utils.youtube_utils import get_video_id, download_audio, download_subtitle
from src.utils.text_utils import clean_subtitle, clean_whisper_output
from src.utils.metrics import compute_bertscore, compute_rouge

def process_single_video(video, whisper_model=None):
    """Process a single YouTube video to generate summary and metrics."""
    try:
        from src.utils.youtube_utils import get_video_info, get_random_delay
        import time
        
        time.sleep(get_random_delay())
        info = get_video_info(video)
        if not info:
            return None
            
        result = generate_summary(info["URL"], whisper_model)
        if not result:
            return None
            
        summary_bert, original_text, lang = result
        
        # Calculate metrics
        P, R, F1 = compute_bertscore(summary_bert, original_text, lang=lang)
        rouge_scores = compute_rouge(summary_bert, original_text)
        
        return (
            info["Title"], info["URL"], info["Duration"],
            info["Author"], info["Views"],
            summary_bert,
            F1, rouge_scores["rouge1"], rouge_scores["rouge2"], rouge_scores["rougeL"],
            lang
        )
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None

def generate_summary(video_url, whisper_model=None):
    """Generate summary for a YouTube video using subtitles or transcription."""
    try:
        import time
        from src.utils.youtube_utils import get_random_delay
        
        time.sleep(get_random_delay())
        video_id = get_video_id(video_url)
        if not video_id:
            return None
            
        text_lines = []
        subtitle_content = None
        detected_lang = None
        
        # Try to get subtitles first
        import yt_dlp
        ydl_sub_opts = {
            'cookiefile': 'cookies.txt',
            'subtitlesformat': 'srt',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True
        }
        
        with yt_dlp.YoutubeDL(ydl_sub_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if info:
                for lang in ['vi', 'en']:
                    if f'{lang}' in info.get('subtitles', {}):
                        subtitle_url = info['subtitles'][lang][0]['url']
                        subtitle_content = download_subtitle(subtitle_url, video_id)
                        if subtitle_content:
                            detected_lang = lang
                            break
                            
        if subtitle_content:
            cleaned = clean_subtitle(subtitle_content)
            if cleaned:
                text_lines = [line.strip() for line in cleaned.split('. ') if len(line.strip()) > 20]
                if not detected_lang:
                    lang_sample = " ".join(text_lines[:3])
                    detected_lang = detect(lang_sample) if lang_sample else None
                    
        # If no subtitles, use Whisper for transcription
        if not text_lines or not detected_lang:
            print(f"No subtitles found, trying audio transcription for video {video_id}")
            audio_path = download_audio(video_url)
            if not audio_path:
                print(f"Failed to download audio for video {video_id}")
                return None
                
            try:
                if whisper_model is None:
                    whisper_model = get_whisper_model()
                    
                # Thay đổi: Không truyền tham số language hoặc truyền None
                segments = transcribe_audio(audio_path)
                
                # Kiểm tra segments có phải là list rỗng không
                if segments and len(segments) > 0:
                    transcript = " ".join([segment.text for segment in segments])
                    
                    if transcript:
                        print(f"Successfully transcribed audio for video {video_id}")
                        cleaned_transcript = clean_whisper_output(transcript)
                        text_lines = [line.strip() for line in cleaned_transcript.split('. ') if len(line.strip()) > 20]
                        lang_sample = " ".join(text_lines[:3])
                        detected_lang = detect(lang_sample) if lang_sample else None
                        print(f"Detected language: {detected_lang}")
                else:
                    print(f"No transcription segments returned for video {video_id}")
            except Exception as e:
                print(f"Whisper transcription error for video {video_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
                
        # Generate summary if we have text and language
        if text_lines and detected_lang in ['vi', 'en']:
            original_text = ". ".join(text_lines)
            summary = summarize_text(original_text, detected_lang)
            return summary, original_text, detected_lang
            
        return None
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None

