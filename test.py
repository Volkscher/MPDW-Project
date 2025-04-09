from datasets import load_dataset
import json

# Video Preparation 

# Load the dataset, trust_remote_code=True is needed to load the dataset from the remote repository.
#dataset = load_dataset('dataset-download.py', trust_remote_code=True) 

with open('activity_net.v1-3.min.json', 'r') as json_data:
    data = json.load(json_data)
    print(data)