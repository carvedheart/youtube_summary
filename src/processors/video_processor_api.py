"""Video processing logic using official YouTube API."""

import os
import re
from langdetect import detect
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from src.models.whisper_model import get_whisper_model
from src.models.summarization_model import summarize_text
from src.utils.youtube_utils import get_video_id
from src.utils.text_utils import clean_subtitle
from src.utils.metrics import compute_bertscore, compute_rouge
from src.utils.oauth_utils import get_authenticated_service
from src.config import YOUTUBE_API_KEY

def get_youtube_service():
    """Initialize the YouTube API service with OAuth 2.0."""
    try:
        # Sử dụng OAuth 2.0 để xác thực
        return get_authenticated_service()
    except Exception as e:
        print(f"Error authenticating with OAuth: {str(e)}")
        print("Falling back to API key authentication for limited access...")
        # Fallback to API key for limited access
        return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def get_video_captions(video_id):
    """Get captions for a video using YouTube API with OAuth 2.0."""
    try:
        # Sử dụng OAuth 2.0 để lấy phụ đề
        youtube = get_youtube_service()
        
        # Lấy danh sách phụ đề
        captions_response = youtube.captions().list(
            part="snippet",
            videoId=video_id
        ).execute()
        
        captions = captions_response.get("items", [])
        if not captions:
            print(f"No captions found for video {video_id}")
            # Fallback to YouTube Transcript API
            return get_video_captions_with_transcript_api(video_id)
        
        # Tìm phụ đề tiếng Anh hoặc tiếng Việt
        caption_id = None
        language = None
        
        for caption in captions:
            lang = caption["snippet"]["language"]
            if lang in ["en", "vi"]:
                caption_id = caption["id"]
                language = lang
                break
        
        if not caption_id:
            print(f"No English or Vietnamese captions found for video {video_id}")
            # Fallback to YouTube Transcript API
            return get_video_captions_with_transcript_api(video_id)
        
        # Tải phụ đề
        caption_response = youtube.captions().download(
            id=caption_id,
            tfmt="srt"
        ).execute()
        
        # Làm sạch phụ đề
        cleaned_captions = clean_subtitle(caption_response.decode("utf-8"))
        
        return cleaned_captions, language
    
    except HttpError as e:
        print(f"YouTube API error: {str(e)}")
        # Fallback to YouTube Transcript API
        return get_video_captions_with_transcript_api(video_id)
    except Exception as e:
        print(f"Error getting captions: {str(e)}")
        # Fallback to YouTube Transcript API
        return get_video_captions_with_transcript_api(video_id)

def get_video_captions_with_transcript_api(video_id):
    """Get captions using YouTube Transcript API as fallback."""
    try:
        print(f"Trying YouTube Transcript API for video {video_id}")
        # Try to get transcript using YouTube Transcript API
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find English or Vietnamese transcript
        transcript = None
        language = None
        
        # First try to get manually created transcripts
        for lang in ['en', 'vi']:
            try:
                transcript = transcript_list.find_transcript([lang])
                language = lang
                break
            except NoTranscriptFound:
                continue
        
        # If no manual transcript, try auto-generated
        if not transcript:
            try:
                transcript = transcript_list.find_transcript(['en'])
                language = 'en'
            except NoTranscriptFound:
                try:
                    transcript = transcript_list.find_transcript(['vi'])
                    language = 'vi'
                except NoTranscriptFound:
                    # Try to get any transcript and translate to English
                    try:
                        transcript = transcript_list.find_transcript([])
                        transcript = transcript.translate('en')
                        language = 'en'
                    except:
                        return None, None
        
        # Get the transcript data
        transcript_data = transcript.fetch()
        
        # Convert transcript data to text
        text_parts = [part['text'] for part in transcript_data]
        full_text = ' '.join(text_parts)
        
        return full_text, language
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"No transcripts available for video {video_id}: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error getting captions with Transcript API: {str(e)}")
        return None, None

def get_video_details(video_id):
    """Get video details using YouTube API."""
    try:
        youtube = get_youtube_service()
        
        # Get video details
        video_response = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        ).execute()
        
        if not video_response["items"]:
            print(f"Video {video_id} not found")
            return None
        
        video = video_response["items"][0]
        
        # Extract relevant information
        return {
            "Title": video["snippet"]["title"],
            "URL": f"https://www.youtube.com/watch?v={video_id}",
            "Duration": video["contentDetails"]["duration"],
            "Author": video["snippet"]["channelTitle"],
            "Views": video["statistics"].get("viewCount", "0"),
            "PublishedAt": video["snippet"]["publishedAt"]
        }
    
    except Exception as e:
        print(f"Error getting video details: {str(e)}")
        return None

def process_video_api(video_id):
    """Process a video using YouTube API."""
    try:
        # Get video details
        video_info = get_video_details(video_id)
        if not video_info:
            return None
        
        # Get captions
        captions, language = get_video_captions(video_id)
        
        if not captions or not language:
            print(f"No captions available for {video_info['Title']}")
            return None
        
        # Generate summary
        summary = summarize_text(captions, language)
        
        # Calculate metrics
        P, R, F1 = compute_bertscore(summary, captions, lang=language)
        rouge_scores = compute_rouge(summary, captions)
        
        # Create result
        result = {
            "Title": video_info["Title"],
            "URL": video_info["URL"],
            "Duration": video_info["Duration"],
            "Author": video_info["Author"],
            "Views": video_info["Views"],
            "Summary": summary,
            "BERTScore_F1": F1,
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "Language": language
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
