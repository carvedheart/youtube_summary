"""Simple YouTube processor without PySpark dependencies."""

import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.config import MAX_VIDEOS_PER_PLAYLIST
from src.models.whisper_model import get_whisper_model
from src.models.summarization_model import summarize_text
from src.utils.youtube_utils import get_video_id
from src.utils.metrics import compute_bertscore, compute_rouge
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytubefix import Playlist
import re
from itertools import islice

def extract_playlist_id(url):
    """Extract playlist ID from URL."""
    import re
    
    # Try to match playlist ID
    playlist_id_match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
    if playlist_id_match:
        return playlist_id_match.group(1)
    
    return None

def get_video_captions(video_id):
    """Get captions using YouTube Transcript API."""
    try:
        print(f"Getting captions for video {video_id}")
        
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find English or Vietnamese transcript
        transcript = None
        language = None
        
        # Try manually created transcripts first
        for lang_code in ['en', 'vi']:
            try:
                transcript = transcript_list.find_transcript([lang_code])
                language = lang_code
                break
            except NoTranscriptFound:
                continue
        
        # If no manual transcript, try auto-generated
        if not transcript:
            try:
                # Get all available transcripts
                available_transcripts = list(transcript_list)
                
                # Try to find English or Vietnamese
                for t in available_transcripts:
                    if t.language_code in ['en', 'vi']:
                        transcript = t
                        language = t.language_code
                        break
                
                # If still no transcript, use the first available and translate to English
                if not transcript and available_transcripts:
                    transcript = available_transcripts[0]
                    transcript = transcript.translate('en')
                    language = 'en'
            except Exception as e:
                print(f"Error finding transcript: {str(e)}")
        
        if not transcript:
            return None, None
        
        # Get transcript data
        transcript_data = transcript.fetch()
        
        # Extract text
        text = " ".join([entry.get('text', '') for entry in transcript_data])
        
        return text, language
    
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"No transcripts available for video {video_id}: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error with YouTube Transcript API: {str(e)}")
        return None, None

def process_video_simple(video_id):
    """Process a single video with simple approach."""
    try:
        from pytubefix import YouTube
        
        # Get video info
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        video_info = {
            "Title": yt.title,
            "URL": f"https://www.youtube.com/watch?v={video_id}",
            "Duration": str(yt.length),
            "Author": yt.author,
            "Views": str(yt.views)
        }
        
        # Get captions
        captions, language = get_video_captions(video_id)
        if not captions:
            print(f"No captions found for video {video_id}")
            return None
        
        # Generate summary
        try:
            summary = summarize_text(captions, language)
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            summary = f"Summary of {video_info['Title']} (auto-generated due to error)"
        
        # Calculate metrics
        try:
            P, R, F1 = compute_bertscore(summary, captions, lang=language)
            rouge_scores = compute_rouge(summary, captions)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            P, R, F1 = 0, 0, 0
            rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
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

def process_youtube_playlist_simple(url, max_videos=MAX_VIDEOS_PER_PLAYLIST):
    """Process a YouTube playlist using a simple approach without PySpark."""
    # Create directories
    os.makedirs("output", exist_ok=True)
    
    try:
        # Initialize playlist
        playlist = Playlist(url)
        playlist._video_regex = re.compile(r'"videoId":"([^"]+)"')
        videos = list(islice(playlist.videos, max_videos))
        
        if not videos:
            print("[ERROR] No videos found in playlist.")
            return None
        
        print(f"Found {len(videos)} videos in playlist.")
        
        # Extract video IDs
        video_ids = [get_video_id(video.watch_url) for video in videos]
        
        # Process videos
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_video_simple, video_id): video_id for video_id in video_ids if video_id}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                video_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"Processed: {result['Title']}")
                except Exception as e:
                    print(f"Error processing video {video_id}: {str(e)}")
        
        if not results:
            print("[WARN] No video processed successfully.")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_path = "output/youtube_summary_results_simple.csv"
        df.to_csv(output_path, index=False)
        print(f"\nProcessed {len(df)} videos successfully")
        print(f"Results saved to {output_path}")
        
        return df
    
    except Exception as e:
        print(f"Error processing playlist: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for the command line."""
    parser = argparse.ArgumentParser(description="Process YouTube playlists (Simple Version)")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max-videos", type=int, default=MAX_VIDEOS_PER_PLAYLIST, 
                        help="Maximum number of videos to process")
    
    args = parser.parse_args()
    process_youtube_playlist_simple(args.url, args.max_videos)

if __name__ == "__main__":
    main()