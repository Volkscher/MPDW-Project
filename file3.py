from datasets import load_dataset

dataset = load_dataset('dataset-download.py', trust_remote_code=True)

print(dataset['train'][0])  # First training example
print(dataset['validation'][0])  # First validation example
print(dataset['test'][0])  # First test example

#print(dataset['train'].features)  # Features of the training set
#print(dataset['validation'].features)  # Features of the validation set

print('train length:', len(dataset['train']))  # Number of training examples
print('test length:', len(dataset['test']))  # Number of validation examples