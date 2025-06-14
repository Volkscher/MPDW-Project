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
import json
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
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

def create_opensearch_index():
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

    index_exists = client.indices.exists(index=index_name)
    if not index_exists:
        client.indices.create(index=index_name, body=index_body)
        print(f"Índice '{index_name}' criado.")
        return True
    else:
        print(f"Índice '{index_name}' já existe.")
        return False
    
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

## Create the OpenSearch client, load the data and index it
def load_phase2():
    # Load the dataset, trust_remote_code=True is needed to load the dataset from the remote repository.
    # This is where we get our captions from.
    dataset = load_dataset('dataset-download.py', trust_remote_code=True) 

    # Load the CLIP Model and Processor for embedding the image and text fields
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    index_number_id = 0  # Index number to use as document ID (0, 1, 2, ...)
    # Load captions and video metadata
    with open("top10.json", "r") as data_file:
        video_metadata = json.load(data_file)

    # Directory containing the extracted keyframes
    keyframes_dir = "keyframes"

    # Iterate over each video in the dataset and store their metadata
    videos_info = {}
    for split in ['train', 'test', 'validation']:
        for video in dataset[split]:
            video['video_id'] = video['video_id'].replace("v_", "")
            
            # Iterate through the documents in doc_list
            videos_info[video['video_id']] = {
                "video_id": video['video_id'],
                "url": video['video_path'],
                "duration": video['duration'],
                "captions_starts": video['captions_starts'],
                "captions_ends": video['captions_ends'],
                "captions": video['en_captions'],
            }
            
            #print(f"Video ID: {video['video_id']}")

    # Iterate over each video
    for video_id, video_info in video_metadata.items():
        # Download the video and extract frames every 2 seconds
        if extract_keyframes(video_url= video_info['url'], output_dir=f"{keyframes_dir}/{video_id}", interval= 2) == False:
            print(f"Failed to extract keyframes for video {video_id}. Skipping...")
            continue
        
        # Get the stored frames of the video
        # keyframes/{video_id}/frame_0000.jpg
        video_keyframes_dir = os.path.join(keyframes_dir, video_id)
        frame_files = sorted(os.listdir(video_keyframes_dir))
        
        # Compute CLIP image/frame embeddings
        # frame_embeddings = <frame_file>: [<embedding>]
        frame_embeddings = compute_clip_embeddings(image_dir = f"{keyframes_dir}/{video_id}")

        for i, frame_file in enumerate(frame_files):
            # Skip if the file is not a valid image
            # This can happen if the video has been privated and therefore no download happenned
            if frame_file not in frame_embeddings:
                print(f"Warning: Frame '{frame_file}' not found in embeddings. Skipping...")
                continue
            
            # Calculate the timestamp range for this frame
            # VERIFY THAT THIS FRAME_DURATION IS WORKING (we might have deleted something that was a variable)
            frame_duration = 2.0
            start_time = i * frame_duration
            end_time = start_time + frame_duration
            
            # Find captions that overlap with this frame's time range
            matching_video_info = videos_info[video_id]
            matching_captions = get_caption_from_timeframe(matching_video_info["captions_starts"], matching_video_info["captions_ends"], matching_video_info["captions"], start_time, end_time)
            
            # Find labels that overlap with this frame's time range
            label = ""
            for annotation in video_info["annotations"]:
                if (annotation["segment"][0] >= start_time and annotation["segment"][1] <= end_time):
                    label = annotation["label"]
            
            # Generate caption embeddings
            inputs = processor(text=matching_captions, return_tensors="pt", padding=True).to(device)
            caption_embeddings = model.get_text_features(**inputs)
            caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)
            caption_embeddings = caption_embeddings.detach().cpu().tolist()
            
            # Generate label embeddings
            inputs = processor(text=label, return_tensors="pt", padding=True).to(device)
            label_embeddings = model.get_text_features(**inputs)
            label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)
            label_embeddings = label_embeddings.detach().cpu().tolist()

            # Check if caption_embeddings is not empty to avoid IndexError
            if not caption_embeddings or not caption_embeddings[0]:
                print(f"Warning: No caption embeddings for frame '{frame_file}' in video '{video_id}'. Skipping...")
                continue

            # Create the OpenSearch document
            doc = {
                "video_id": video_id,
                "frame_file": frame_file,
                "start_time": start_time,
                "end_time": end_time,
                "captions": matching_captions,
                "label": label,
                "label_embeddings": label_embeddings[0],
                "frame_embedding": frame_embeddings[frame_file][0],
                "caption_embeddings": caption_embeddings[0],
                "video_path": video_info["url"],
                "duration": video_info["duration"]
            }
            
            # Check if this is really needed
            if doc['duration'] is None:
                print(f"Warning: Duration for video '{video_id}' is None. Skipping...")
                continue
            
            # Index the document
            try:
                response = client.index(index=index_name, id=index_number_id, body=doc)
                print(f"Indexed document {index_number_id}: {response}")
                index_number_id += 1
            except Exception as e:
                print(f"Error indexing document {index_number_id}: {e}")

    # Refresh the index so the docs are searchable
    client.indices.refresh(index = 'user13')

def setup_phase2():
    index_created = create_opensearch_index()

    if (not index_created):
        load_phase2()
    else:
        load_phase2()
        # index already created, we don't download or load anything, to do later

