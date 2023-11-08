

import logging
import math
import os
import sys
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util, InputExample
from torch.utils.data import DataLoader

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'mteb/sickr-sts.tsv.gz'
# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://huggingface.co/datasets/mteb/sickr-sts.tsv.gz', sts_dataset_path)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilbert-base-uncased'
# Read the dataset
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_' + model_name.replace("/", "-") + '-' + datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S")
# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
dataset = load_dataset(path='sickr-sts', data_files='test.jsonl')
# Get the number of samples in the dataset
num_samples = len(dataset['train'])
# Calculate the split sizes
split_ratio = 0.6  # 60% for training, 40% for testing
train_size = int(num_samples * split_ratio)
test_size = num_samples - train_size
# Create a list of indices and shuffle them
indices = list(range(num_samples))
# Split the indices into training and testing subsets
train_indices = indices[:train_size]
test_indices = indices[train_size:]
train_examples = []
for sample in dataset['train'].select(train_indices):
    train_examples.append(InputExample(texts=[sample['sentence1'], sample['sentence2']], label=sample['score']))

# Create DataLoader instances for training and testing
train_dataloader = DataLoader(train_examples, shuffle=False, batch_size=1)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

