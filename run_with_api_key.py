"""Run YouTube processor using only YouTube API Key without OAuth."""

import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import xml.etree.ElementTree
import xml.parsers.expat

from src.config import YOUTUBE_API_KEY, MAX_VIDEOS_PER_PLAYLIST
from src.models.summarization_model import summarize_text
from src.utils.metrics import compute_bertscore, compute_rouge
from src.utils.youtube_utils import get_video_id
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_youtube_service():
    """Initialize the YouTube API service with API key only."""
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def get_playlist_videos(playlist_id, max_videos=MAX_VIDEOS_PER_PLAYLIST):
    """Get videos from a playlist using YouTube API with API key."""
    youtube = get_youtube_service()
    
    # Get playlist details
    playlist_response = youtube.playlists().list(
        part="snippet",
        id=playlist_id
    ).execute()
    
    if not playlist_response["items"]:
        print(f"Playlist {playlist_id} not found")
        return None, []
    
    playlist_title = playlist_response["items"][0]["snippet"]["title"]
    
    # Get playlist items
    video_ids = []
    next_page_token = None
    
    while len(video_ids) < max_videos:
        playlist_items_response = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        
        for item in playlist_items_response["items"]:
            if item["snippet"]["resourceId"]["kind"] == "youtube#video":
                video_ids.append(item["snippet"]["resourceId"]["videoId"])
                if len(video_ids) >= max_videos:
                    break
        
        next_page_token = playlist_items_response.get("nextPageToken")
        if not next_page_token:
            break
    
    return playlist_title, video_ids

def extract_playlist_id(url):
    """Extract playlist ID from URL."""
    import re
    
    # Try to match playlist ID
    playlist_id_match = re.search(r"list=([a-zA-Z0-9_-]+)", url)
    if playlist_id_match:
        return playlist_id_match.group(1)
    
    return None

def get_video_details(video_id):
    """Get video details using YouTube API with API key."""
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
        try:
            transcript_data = transcript.fetch()
        except (xml.etree.ElementTree.ParseError, xml.parsers.expat.ExpatError) as e:
            print(f"XML parsing error for video {video_id}: {str(e)}")
            return None, None
        
        # Extract text
        try:
            if isinstance(transcript_data, list):
                text = " ".join([entry.get('text', '') for entry in transcript_data])
            else:
                if hasattr(transcript_data, 'text'):
                    text = transcript_data.text
                elif hasattr(transcript_data, 'get_text'):
                    text = transcript_data.get_text()
                elif hasattr(transcript_data, '__str__'):
                    text = str(transcript_data)
                else:
                    print(f"Unknown transcript format: {type(transcript_data)}")
                    return None, None
        except Exception as e:
            print(f"Error extracting text from transcript: {str(e)}")
            return None, None
        
        return text, language
    
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"No transcripts available for video {video_id}: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Error with YouTube Transcript API: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def process_video(video_id):
    """Process a single YouTube video using API Key."""
    try:
        # Get video details
        video_info = get_video_details(video_id)
        if not video_info:
            print(f"Could not get details for video {video_id}")
            return None
            
        # Get captions
        captions, language = get_video_captions(video_id)
        if not captions:
            print(f"No captions found for video {video_id}")
            return None
            
        print(f"Processing video: {video_info['Title']} (language: {language})")
        
        # Generate summary
        try:
            summary = summarize_text(captions, language)
            if not summary or summary.startswith("Could not generate summary"):
                print(f"Failed to generate summary for {video_info['Title']}")
                # Tạo một tóm tắt đơn giản
                summary = f"Summary of {video_info['Title']} (auto-generated due to error)"
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

def process_youtube_playlist_api_key(url, max_videos=MAX_VIDEOS_PER_PLAYLIST):
    """Process a YouTube playlist using YouTube API with API key."""
    # Create directories
    os.makedirs("output", exist_ok=True)
    
    try:
        # Extract playlist ID
        playlist_id = extract_playlist_id(url)
        if not playlist_id:
            print("[ERROR] Invalid playlist URL")
            return None
        
        # Get playlist videos
        playlist_title, video_ids = get_playlist_videos(playlist_id, max_videos)
        if not video_ids:
            print("[ERROR] No videos found in playlist")
            return None
        
        print(f"Playlist: {playlist_title}")
        print(f"Found {len(video_ids)} videos")
        
        # Process videos with improved parallelism
        results = []
        # Increase max_workers for better parallelism
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks at once
            futures = {executor.submit(process_video, video_id): video_id for video_id in video_ids}
            
            # Process results as they complete
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
            print("[WARN] No video processed successfully")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        output_path = "output/youtube_summary_results_api_key.csv"
        df.to_csv(output_path, index=False)
        print(f"\nProcessed {len(df)} videos successfully")
        print(f"Results saved to {output_path}")
        
        # Try to upload to MongoDB if configured
        try:
            from src.storage.mongodb_handler_pandas import upload_to_mongodb_pandas
            upload_to_mongodb_pandas(df)
            print("Results uploaded to MongoDB")
        except Exception as e:
            print(f"Error uploading to MongoDB: {str(e)}")
        
        return df
    
    except Exception as e:
        print(f"Error processing playlist: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for the command line."""
    parser = argparse.ArgumentParser(description="Process YouTube playlists using API Key")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max-videos", type=int, default=MAX_VIDEOS_PER_PLAYLIST, 
                        help="Maximum number of videos to process")
    
    args = parser.parse_args()
    process_youtube_playlist_api_key(args.url, args.max_videos)

if __name__ == "__main__":
    main()







