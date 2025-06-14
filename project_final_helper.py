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
        print(f"Index '{index_name}' created.")
        return True
    else:
        print(f"Index '{index_name}' already exists. Skipping creation and indexing...")
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
            saved_count += 1
        
        frame_count += 1
        
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
    matching_captions = [""]
    
    for i, caption in enumerate(en_captions):
        if captions_starts[i] <= f_start_time and captions_ends[i] >= f_end_time:
            matching_captions.append(caption)
    
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
            # Frame duration is 2, since that is what is present in extract_keyframes
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

def search_frame_by_text(text_query, top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    text_tokens = clip.tokenize([text_query]).to(device)
    
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).cpu().numpy()[0]

    query_body = {
        "size": 5,
        "query": {
            "knn": {
                "frame_embedding": {
                    "vector": text_embedding.tolist(),
                    "k": top_k,
                }
            }
        }
    }
    
    response = client.search(index="user13", body=query_body)
    to_return = []
    
    for hit in response['hits']['hits']:
        file_hit = {
            "video_id": hit['_source']['video_id'],
            "frame_path": hit['_source']['frame_file'],
            "score": hit['_score'],
            "captions": hit['_source']['captions']
        }

        to_return.append(file_hit)
    
    return to_return

def search_caption_by_image(image_path, top_k=5):
    to_return = []
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Embed the image passed by the user
    # With this we can search for the most similar caption to the image
    # This works because they are embedded with the same model, and therefore in the same space, which means we can compare them
    img = Image.open(image_path)
    img = preprocess(img).unsqueeze(0).to(device)
    img_embedding = model.encode_image(img).cpu().tolist()[0]

    query_body = {
        "size": 5,
        "query": {
            "knn": {
                "caption_embeddings": {
                    "vector": img_embedding,
                    "k": top_k,
                }
            }
        }
    }
    
    response = client.search(index="user13", body=query_body)
    
    # Show the Image
    for hit in response['hits']['hits']:
        file_hit = {
            "video_id": hit['_source']['video_id'],
            "frame_path": hit['_source']['frame_file'],
            "score":hit['_score'],
            "captions": hit['_source']['captions']
        }

        to_return.append(file_hit)
        
    return to_return

def search_frame_by_image(image_path, top_k=5):
    to_return = []
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Embed the image passed by the user
    # With this we can search for the most similar caption to the image
    # This works because they are embedded with the same model, and therefore in the same space, which means we can compare them
    img = Image.open(image_path)
    img = preprocess(img).unsqueeze(0).to(device)
    img_embedding = model.encode_image(img).cpu().tolist()[0]

    query_body = {
        "size": 5,
        "query": {
            "knn": {
                "frame_embedding": {
                    "vector": img_embedding,
                    "k": top_k,
                }
            }
        }
    }
    
    response = client.search(index="user13", body=query_body)

    for hit in response['hits']['hits']:
        file_hit = {
            "video_id": hit['_source']['video_id'],
            "frame_path": hit['_source']['frame_file'],
            "score": hit['_score'],
            "captions": hit['_source']['captions']
        }

        to_return.append(file_hit)
    
    return to_return
    

def display_comparison(filename, hit):
    # Load query and retrieved images
    query_img = Image.open(f"./sample_images/{filename}") # query image
    hit_img = Image.open(f"./keyframes/{hit['video_id']}/{hit['frame_path']}") #

    # Create side-by-side plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Query image (the image the user inserted)
    axs[0].imshow(query_img)
    axs[0].set_title("Query Frame")
    axs[0].axis('off')

    # Retrieved image (the hit)
    axs[1].imshow(hit_img)
    axs[1].set_title(f"Retrieved (Score: {hit['score']:.4f})")
    axs[1].axis('off')

    # Add caption below both images
    caption_text = "\n".join(hit['captions']) if isinstance(hit['captions'], list) else "no captions"
    fig.text(0.5, 0.02, caption_text, ha='center', wrap=True, fontsize=10)

    plt.tight_layout()
    plt.show()
    

    

def setup_phase2():
    index_created = create_opensearch_index()

    if (index_created): #if the index was created, then there was no previous index. Therefore we need to load and index the documents
        load_phase2()
