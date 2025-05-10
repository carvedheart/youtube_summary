"""Playlist processing logic for YouTube playlists."""

import re
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from pytubefix import Playlist
from pyspark.sql import SparkSession
from src.config import MAX_VIDEOS_PER_PLAYLIST, BATCH_SIZE
from src.models.whisper_model import get_whisper_model
from src.processors.video_processor import process_single_video
from src.utils.youtube_utils import channel_or_playlist

def process_playlist(url, spark, max_videos=MAX_VIDEOS_PER_PLAYLIST):
    """Process a YouTube playlist and return a DataFrame with results."""
    if channel_or_playlist(url) != "Playlist":
        print("[ERROR] Only playlist supported.")
        return None
        
    # Initialize playlist
    playlist = Playlist(url)
    playlist._video_regex = re.compile(r'"videoId":"([^"]+)"')
    videos = list(islice(playlist.videos, max_videos))
    
    if not videos:
        print("[ERROR] No videos found in playlist.")
        return None
        
    # Initialize Whisper model (shared across threads)
    whisper_model = get_whisper_model()
    
    # Process videos in batches
    batch_size = min(BATCH_SIZE, len(videos))
    video_batches = [videos[i:i + batch_size] for i in range(0, len(videos), batch_size)]
    
    results = []
    for batch in video_batches:
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = []
            for video in batch:
                futures.append(executor.submit(process_single_video, video, whisper_model))
                
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing video: {str(e)}")
                    
    if not results:
        print("[WARN] No video processed successfully.")
        return None
        
    # Create DataFrame from results
    return spark.createDataFrame(results, [
        "Title", "URL", "Duration", "Author", "Views", "Summary_BERT",
        "BERTScore_F1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Language"
    ])