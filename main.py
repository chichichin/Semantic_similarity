import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import util
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def scale_number(original_value, min_original, max_original, min_new, max_new):
    scaled_value = ((original_value - min_original) / (max_original - min_original)) * (max_new - min_new) + min_new
    return scaled_value


tokenizer = AutoTokenizer.from_pretrained('all-MiniLM-L12-v2')
model = AutoModel.from_pretrained('all-MiniLM-L12-v2')

dataset = load_dataset(path='sickr-sts', data_files='test.jsonl')

# Get the number of samples in the dataset
num_samples = len(dataset['train'])

# Calculate the split sizes
split_ratio = 0.6  # 60% for training, 40% for testing
train_size = int(num_samples * split_ratio)
test_size = num_samples - train_size

# Create a list of indices and shuffle them
indices = list(range(num_samples))
# random.shuffle(indices)

# Split the indices into training and testing subsets
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create DataLoader instances for training and testing
train_dataloader = DataLoader(dataset['train'].select(train_indices), shuffle=False, batch_size=1)
test_dataloader = DataLoader(dataset['train'].select(test_indices), shuffle=False, batch_size=1)

result = []


def process_data(data):
    sentences = [data['sentence1'][0], data['sentence2'][0]]
    label = float(data['score'])
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    score = util.dot_score(sentence_embeddings[0], sentence_embeddings[1])
    return {
        'sentence1': data['sentence1'],
        'sentence2': data['sentence2'],
        'predicted_score': float(score),
        'score': label
    }


num_threads = 2
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    result = list(tqdm(executor.map(process_data, train_dataloader), total=len(train_dataloader)))
df = pd.DataFrame(result)
score, p = spearmanr(df['predicted_score'], df['score'])
print(score, p)
df.to_csv('result_spearmanr.csv')
