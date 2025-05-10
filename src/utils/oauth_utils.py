"""Utility functions for OAuth 2.0 authentication with YouTube API."""

import os
import pickle
import webbrowser
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError

# Định nghĩa phạm vi (scope) cho YouTube API
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_PICKLE_FILE = "token.pickle"
CLIENT_SECRETS_FILE = "client_secrets.json"

def get_authenticated_service():
    """Get an authenticated YouTube API service using OAuth 2.0."""
    credentials = None
    
    # Kiểm tra xem đã có token được lưu trước đó chưa
    if os.path.exists(TOKEN_PICKLE_FILE):
        print("Loading credentials from file...")
        with open(TOKEN_PICKLE_FILE, "rb") as token:
            credentials = pickle.load(token)
    
    # Nếu không có credentials hoặc credentials hết hạn
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                print("Refreshing access token...")
                credentials.refresh(Request())
            except RefreshError:
                print("Token refresh failed. Getting new token...")
                credentials = None
                # Xóa token cũ nếu không thể làm mới
                if os.path.exists(TOKEN_PICKLE_FILE):
                    os.remove(TOKEN_PICKLE_FILE)
        
        if not credentials:
            print("Fetching new tokens...")
            # Kiểm tra xem file client_secrets.json có tồn tại không
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise FileNotFoundError(
                    f"Client secrets file not found: {CLIENT_SECRETS_FILE}. "
                    "Please download it from Google Cloud Console."
                )
            
            # Tạo flow OAuth 2.0 từ file client_secrets.json
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            
            # Hiển thị thông báo hướng dẫn
            print("\n" + "="*80)
            print("Một cửa sổ trình duyệt sẽ mở ra để bạn đăng nhập và cấp quyền.")
            print("Nếu bạn gặp lỗi 'Đã chặn quyền truy cập', hãy:")
            print("1. Đảm bảo bạn đã thêm email của mình vào danh sách người dùng thử nghiệm")
            print("   trong Google Cloud Console > APIs & Services > OAuth consent screen")
            print("2. Nếu bạn thấy nút 'Advanced', hãy nhấn vào đó và chọn 'Go to [App Name] (unsafe)'")
            print("="*80 + "\n")
            
            # Mở trình duyệt để người dùng đăng nhập và cấp quyền
            credentials = flow.run_local_server(port=8080)
        
        # Lưu credentials để sử dụng lần sau
        with open(TOKEN_PICKLE_FILE, "wb") as token:
            pickle.dump(credentials, token)
    
    # Xây dựng dịch vụ YouTube API
    return build("youtube", "v3", credentials=credentials)
