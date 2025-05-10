"""Utility functions for YouTube video and playlist processing."""

import re
import time
import random
import yt_dlp
import os
import json
import requests
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from fake_useragent import UserAgent
from src.config import YOUTUBE_API_KEY

def get_youtube_service():
    """Initialize the YouTube API service."""
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def get_random_delay():
    """Get a random delay to avoid rate limiting."""
    return random.uniform(1, 5)

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    if "youtube.com/watch" in url:
        parsed_url = urlparse(url)
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_video_info(video):
    """Get metadata for a YouTube video."""
    try:
        time.sleep(random.uniform(1, 3))
        ydl_opts = {
            'cookiefile': 'cookies.txt',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ignoreerrors': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video.watch_url, download=False)
            return {
                "Title": info.get('title'),
                "URL": video.watch_url,
                "Duration": format_duration(info.get('duration', 0)),
                "Author": info.get('uploader'),
                "Views": format_views(info.get('view_count', 0))
            }
    except Exception as e:
        print(f"Lỗi khi xử lý video {video.watch_url}: {str(e)}")
        return None

def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS."""
    if not seconds:
        return "Unknown"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def format_views(views):
    """Format view count with K, M, B suffixes."""
    if not views:
        return "Unknown"
    if views < 1000:
        return str(views)
    elif views < 1000000:
        return f"{views/1000:.1f}K"
    elif views < 1000000000:
        return f"{views/1000000:.1f}M"
    else:
        return f"{views/1000000000:.1f}B"

def download_audio(video_url, cache_dir="cache"):
    """Download audio from YouTube video."""
    video_id = get_video_id(video_url)
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, f"audio_{video_id}.mp3")
    audio_extensions = ['mp3', 'm4a', 'webm', 'mp4']
    
    # Check if already cached
    for ext in audio_extensions:
        temp_path = os.path.join(cache_dir, f"audio_{video_id}.{ext}")
        if os.path.exists(temp_path):
            print(f"Using cached audio file: {temp_path}")
            if ext != 'mp3':
                try:
                    os.rename(temp_path, cached_path)
                    return cached_path
                except:
                    return temp_path
            return temp_path
    
    # Check ffmpeg installation
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print(f"Found ffmpeg: {result.stdout.splitlines()[0]}")
    except Exception as e:
        print(f"Warning: ffmpeg check failed: {str(e)}")
    
    # Download if not cached
    print(f"Downloading audio for video {video_id}")
    
    # First try with simple format (no conversion needed)
    try:
        simple_path = os.path.join(cache_dir, f"audio_{video_id}.webm")
        with yt_dlp.YoutubeDL({
            'format': 'bestaudio/best',
            'outtmpl': simple_path,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True
        }) as ydl:
            ydl.download([video_url])
            if os.path.exists(simple_path):
                print(f"Downloaded audio to {simple_path}")
                return simple_path
    except Exception as e:
        print(f"Simple download failed: {str(e)}")
    
    # If that fails, try with more options
    try:
        print("Trying alternative download method...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(cache_dir, 'audio_%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
            # Check if any audio file was created
            for ext in audio_extensions:
                temp_path = os.path.join(cache_dir, f"audio_{video_id}.{ext}")
                if os.path.exists(temp_path):
                    print(f"Downloaded audio to {temp_path}")
                    return temp_path
    except Exception as e:
        print(f"Alternative download failed: {str(e)}")
    
    # Last resort - try to download any format
    try:
        print("Trying last resort download method...")
        fallback_path = os.path.join(cache_dir, f"audio_{video_id}.raw")
        with yt_dlp.YoutubeDL({
            'format': 'worstaudio/worst',
            'outtmpl': fallback_path,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True
        }) as ydl:
            ydl.download([video_url])
            if os.path.exists(fallback_path):
                print(f"Downloaded audio to {fallback_path}")
                return fallback_path
    except Exception as e:
        print(f"Last resort download failed: {str(e)}")
    
    print(f"All download methods failed for video {video_id}")
    return None

def speed_to_str(speed):
    """Convert download speed to human-readable format."""
    if isinstance(speed, (int, float)):
        if speed > 1024*1024:
            return f"{speed/(1024*1024):.2f} MB/s"
        elif speed > 1024:
            return f"{speed/1024:.2f} KB/s"
        else:
            return f"{speed:.2f} B/s"
    return "N/A"

def download_subtitle(url, video_id=None, cache_dir="cache"):
    """Download subtitle from URL and cache it."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if already cached
    if video_id:
        cache_path = f"{cache_dir}/subs_{video_id}.txt"
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
    
    # Download if not cached
    try:
        ua = UserAgent()
        headers = {'User-Agent': ua.random, 'Accept-Language': 'en-US,en;q=0.9'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        text = response.text.strip()
        
        # Handle JSON format subtitles
        if text.startswith('{') or text.startswith('['):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and 'events' in data:
                    captions = []
                    for event in data['events']:
                        if 'segs' in event:
                            for seg in event['segs']:
                                if 'utf8' in seg:
                                    captions.append(seg['utf8'])
                    text = " ".join(captions)
            except Exception:
                pass
        
        # Cache the result
        if video_id:
            with open(f"{cache_dir}/subs_{video_id}.txt", "w", encoding="utf-8") as f:
                f.write(text)
        
        return text
    except Exception as e:
        print(f"[ERROR] Lỗi tải phụ đề: {str(e)}")
        return None

def clean_subtitle(text):
    """Clean subtitle text by removing timestamps and formatting."""
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = ' '.join(text.split())
    return text.strip()

def channel_or_playlist(url):
    """Determine if URL is for a channel or playlist."""
    if "youtube.com/playlist" in url or "list=" in url:
        return "Playlist"
    elif "youtube.com/channel/" in url or "youtube.com/c/" in url or "youtube.com/user/" in url:
        return "Channel"
    else:
        return "Unknown"

