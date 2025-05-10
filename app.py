"""Streamlit app for YouTube Processor demo."""

import streamlit as st
import pandas as pd
import os
import time
from PIL import Image
import base64
from io import BytesIO

from src.main import process_youtube_playlist
from run_with_api_key import process_youtube_playlist_api_key
from simple_processor import process_youtube_playlist_simple

# Set page config
st.set_page_config(
    page_title="YouTube Processor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #505050;
        margin-bottom: 20px;
    }
    .result-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .video-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        flex: 1;
        min-width: 100px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4B5563;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def display_results(df):
    """Display processing results in a nice format."""
    if df is None or len(df) == 0:
        st.error("No results to display")
        return
    
    # Display summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Videos Processed", len(df))
    with col2:
        languages = df["Language"].value_counts().to_dict()
        lang_str = ", ".join([f"{k}: {v}" for k, v in languages.items()])
        st.metric("Languages", lang_str)
    with col3:
        avg_bertscore = df["BERTScore_F1"].mean()
        st.metric("Avg. BERTScore", f"{avg_bertscore:.2f}")
    
    # Display each video result
    for _, row in df.iterrows():
        with st.expander(f"📹 {row['Title']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**URL:** [{row['URL']}]({row['URL']})")
                st.markdown(f"**Author:** {row['Author']}")
                st.markdown(f"**Duration:** {row['Duration']}")
                st.markdown(f"**Views:** {row['Views']}")
                st.markdown(f"**Language:** {row['Language']}")
                
                st.markdown("### Summary")
                st.markdown(row["Summary"])
            
            with col2:
                st.markdown("### Metrics")
                st.markdown(f"**BERTScore F1:** {row['BERTScore_F1']:.4f}")
                st.markdown(f"**ROUGE-1:** {row['ROUGE-1']:.4f}")
                st.markdown(f"**ROUGE-2:** {row['ROUGE-2']:.4f}")
                st.markdown(f"**ROUGE-L:** {row['ROUGE-L']:.4f}")

# Main app
def main():
    st.markdown("<h1 class='main-header'>YouTube Processor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Process YouTube playlists and generate summaries</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=100)
    st.sidebar.title("Settings")

    max_videos = st.sidebar.slider(
        "Maximum Videos to Process",
        min_value=1,
        max_value=30,
        value=5,
        help="Maximum number of videos to process from the playlist"
    )

    # Main content
    playlist_url = st.text_input("Enter YouTube Playlist URL", 
                                placeholder="https://www.youtube.com/playlist?list=...")

    process_button = st.button("Process Playlist", type="primary")

    if process_button and playlist_url:
        with st.spinner("Processing playlist... This may take a few minutes."):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing YouTube API...")
                progress_bar.progress(10)
                
                # Process with API Key
                results = process_youtube_playlist_api_key(playlist_url, max_videos)
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Display results
                if results is not None:
                    st.success(f"Successfully processed {len(results)} videos!")
                    display_results(results)
                else:
                    st.error("Failed to process playlist. Check the console for errors.")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()


