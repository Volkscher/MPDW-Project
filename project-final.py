import urllib
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import clip
import torch
from opensearchpy import OpenSearch
import os
import yt_dlp
import cv2
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_opensearch_index():
    host = 'api.novasearch.org'
    port = 443
    user = 'user13' 
    password = 'rumoao+20' 
    index_name = user # We can only have an index with the same name has our user name.
    
    # Create the client with SSL/TLS enabled, but hostname verification disabled.
    client = OpenSearch(
        hosts = [{'host': host, 'port': port}],
        http_compress = True, # enables gzip compression for request bodies
        http_auth = (user, password),
        use_ssl = True,
        url_prefix = 'opensearch_v2',
        verify_certs = False,
        ssl_assert_hostname = False,
        ssl_show_warn = False
    )

    # Create the index with knn_vector mapping
    index_body = {
        "settings":{
        "index":{
            "number_of_replicas":0,
            "number_of_shards":4,
            "refresh_interval":"1s",
            "knn":"true"
        }
    },
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "frame_file": {"type": "keyword"},
                "label": {"type": "keyword"},
                "start_time": {"type": "float"},
                "end_time": {"type": "float"},
                "captions": {"type": "text"},
                "frame_embedding": {
                    "type": "knn_vector",
                    "dimension": 512,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                },
                "label_embedding": {
                    "type": "knn_vector",
                    "dimension": 512,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                },
                "caption_embeddings": {
                    "type": "knn_vector",
                    "dimension": 512,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                },
                "video_path": {"type": "keyword"},
                "duration": {"type": "float"}
            }
        }
    }

    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)
        print(f"Índice '{index_name}' criado.")
    else:
        print(f"Índice '{index_name}' já existe.")
    
#######

def download_video(youtube_url, output_dir="videos"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set download options
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,  # Show download progress
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_file = os.path.join(output_dir, f"{info['title']}.{info['ext']}")
            #print(f"Downloaded: {video_file}")
            return video_file

    except yt_dlp.utils.DownloadError as e:
        print(f"Failed to download {youtube_url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error with {youtube_url}: {e}")
        return None
    
#########

def extract_keyframes(video_url, output_dir, interval=2):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video_path = download_video(video_url, output_dir= "videos")
    
    if video_path is None:
        # Video has most likely been deleted or privated
        return False
    
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    frame_interval = int(frame_rate * interval)  # Convert to frame interval

    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame every 'interval' seconds
        if frame_count % frame_interval == 0:
            frame_file = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_file, frame)
            #print(f"Saved: {frame_file}")
            saved_count += 1
        
        frame_count += 1
        
    #print(f"Extracted {saved_count} keyframes from '{video_path}' to '{output_dir}'")
    cap.release()
    return True

###############
    
# Computes CLIP embeddings for all images in a directory and returns them in a dictionary with filenames as keys
def compute_clip_embeddings(image_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    embeddings = {}

    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                normalized = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                embeddings[filename] = normalized.cpu().tolist()

    return embeddings

######################

# This function is used to get the captions that are within the timeframe of the video.
def get_caption_from_timeframe(captions_starts:[], captions_ends:[], en_captions: [], f_start_time: float, f_end_time:float):
    #matching_starts = []
    #matching_ends = []
    matching_captions = [""]
    
    for i, caption in enumerate(en_captions):
        print(f"Caption {caption} starts at {captions_starts[i]} and ends at {captions_ends[i]}")
        print(f"Frame starts at {f_start_time} and ends at {f_end_time}")
        if captions_starts[i] <= f_start_time and captions_ends[i] >= f_end_time:
            #matching_starts.append(captions_starts[i])
            #matching_ends.append(captions_ends[i])
            matching_captions.append(caption)
            print(f"Caption {caption} matches the timeframe from {f_start_time} to {f_end_time}")
    
    return matching_captions

