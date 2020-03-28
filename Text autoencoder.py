# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
from tqdm import tqdm_notebook, tqdm

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import torchtext
import collections

# %% [markdown]
# ### Verify if CUDA is available

# %%
# If CUDA is available print devices
if torch.cuda.is_available():
    print('CUDA devices:')
    for device in range(0, torch.cuda.device_count()):
        print('\t{} - {}'.format(device, torch.cuda.get_device_name(device)))
else:
    print('No CUDA devices')

# %% [markdown]
# ### Define encoder and decoder

# %%
class Encoder(nn.Module):
  def __init__(self, embedding_size, encoding_size, use_cuda=False):
    super(Encoder, self).__init__()

    self.encoding_size = encoding_size
   
    self.encoder = nn.LSTM(
      input_size=embedding_size, 
      hidden_size=encoding_size,
      num_layers=1,
      bias=True,
      batch_first=True,
      dropout=0.1,
      bidirectional=False
    )
  
  def forward(self, embeddings):
    _, (hidden, memory) = self.encoder(embeddings)
    return hidden, memory


# %%
class Decoder(nn.Module):

  def __init__(self, encoding_size, embedding_size, use_cuda=False):
    super(Decoder, self).__init__()


    self.encoding_size = encoding_size

    self.decoder = nn.LSTM(
      input_size=embedding_size, 
      hidden_size=encoding_size,
      num_layers=1,
      bias=True,
      batch_first=True,
      dropout=0.1,
      bidirectional=False
    )

    self.dim_linear = nn.Linear(encoding_size, embedding_size)
    self.dim_fn = nn.Tanh()    

  def forward(self, embeddings, init_hidden, init_memory):

    output_, (_, _) = self.decoder(embeddings, (init_hidden, init_memory))
    linear = self.dim_linear(output_)
    
    return self.dim_fn(linear)

# %% [markdown]
# ### Defining functions related with transforming data

# %%
# Transform each sentence by adding a start and finish sequence for encoder and decoder training
def get_transformation(vocabulary, embedding_size):
    first_embedding = torch.ones((embedding_size,))
    last_embedding = torch.zeros((embedding_size,))

    def transform_example(example):
        transformed = []
        transformed.append(first_embedding)

        for idx in example:
            transformed.append(vocabulary.vectors[idx])
        
        transformed.append(last_embedding)
        
        transformed = torch.stack(transformed)[:15]
        transformed = transformed.unsqueeze(0)

        return transformed

    return transform_example


# %%
# Get indices from word's vectors
def get_index_fn(vectors):
    def get_index(prediction):
        indices = []

        for vector in prediction:
            result = torch.abs(vectors - vector).norm(2, dim=1)
            indices.append(torch.argmin(result))

        indices = torch.stack(indices)
        return indices

    return get_index


# %%
# This function recover a sentence from word's indices
def get_text_fn(itos):
    def get_text(example):
        text = []
        for idx in example:
            text.append(itos[idx])

        return ' '.join(text)
    return get_text

# %% [markdown]
# ### Define variables related with loading information and training/validation data

# %%
VECTORS_LOADED = 20000
VAL_PARTITION = 0.05 

# %% [markdown]
# ### Loading word vectors and trainig/validation dataset

# %%
fasttext = torchtext.vocab.FastText(language='en', max_vectors=VECTORS_LOADED)


# %%
vocabulary = torchtext.vocab.Vocab(collections.Counter(fasttext.stoi.keys()))
vocabulary.set_vectors(fasttext.stoi, fasttext.vectors, fasttext.dim)


# %%
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3, vocab=vocabulary)
test_dataset = test_dataset[:int(len(test_dataset) * VAL_PARTITION)]

# %% [markdown]
# ### Defining variables related with training

# %%
USE_CUDA = torch.cuda.is_available()
EMBEDDING_SIZE = fasttext.dim
ENCODING_SIZE = 1024

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 5

# %% [markdown]
# ### Defining training components

# %%
encoder = Encoder(EMBEDDING_SIZE, ENCODING_SIZE, use_cuda=USE_CUDA)
decoder = Decoder(ENCODING_SIZE, EMBEDDING_SIZE, use_cuda=USE_CUDA)

encoder.load_state_dict(torch.load('checkpoints/encoder.pt'))
decoder.load_state_dict(torch.load('checkpoints/decoder.pt'))

if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()


# %%
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=LEARNING_RATE
)

loss_fn = nn.MSELoss()


# %%
transformation_fn = get_transformation(vocabulary, EMBEDDING_SIZE)
get_text = get_text_fn(vocabulary.itos)
get_indices = get_index_fn(vocabulary.vectors)

# %% [markdown]
# ### Training and validation

# %%
def train_step(encoder, decoder, loss_fn, optimizer, batch, use_cuda):
    encoder.train()
    decoder.train()

    if use_cuda:
      batch = batch.cuda()

    encoder.zero_grad()
    decoder.zero_grad()
    optimizer.zero_grad()

    representation, memory = encoder(batch[:,1:,:])

    if use_cuda:
      representation = representation.cuda()
      memory = memory.cuda()
    
    decodings = decoder(batch[:,:-1,:], representation, memory)

    loss = loss_fn(batch[:,1:,:], decodings)
    loss.backward()
    optimizer.step()

    return loss.item()
    


# %%
def val_step(encoder, decoder, loss_fn, batch, use_cuda):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        if use_cuda:
            batch = batch.cuda()

        representation, memory = encoder(batch[:,1:,:])

        if use_cuda:
            representation = representation.cuda()
            memory = memory.cuda()
        
        decodings = decoder(batch[:,:-1,:], representation, memory)

        loss = loss_fn(batch[:,1:,:], decodings)

        return loss.item(), decodings


# %%
# Define steps where examples will be sampled 
example_step = VAL_PARTITION * len(train_dataset)
test_examples = iter(test_dataset)

last_val_loss = None

# For EPOCHS
for epoch in range(EPOCHS):

  # Restart train and validation datasets
  examples = iter(train_dataset)
  val_examples = iter(test_dataset)

  # Progress bar for training dataset
  progress_bar = tqdm(range(len(train_dataset)))
  train_loss = 0
  
  # For all data in training dataset
  
  for batch_idx in progress_bar:

    # Add train loss to progress bar
    progress_bar.set_description('Loss: {}'.format(train_loss / (batch_idx + 1)))
    
    # Train step
    example = next(examples)[1]
    embeddings = transformation_fn(example)

    train_loss += train_step(encoder, decoder, loss_fn, optimizer, embeddings, USE_CUDA)


    if batch_idx % example_step == 0:
      with torch.no_grad():
        try:
          example = next(test_examples)[1]
        except:
          test_examples = iter(test_dataset)
          example = next(test_examples)[1]

        embeddings = transformation_fn(example)

        _, decodings = val_step(encoder, decoder, loss_fn, embeddings, USE_CUDA)

        print('\nReal: {}'.format(get_text(example[:15])))
        print('Decoded: {}'.format(get_text(get_indices(decodings[0].cpu()))))      


  with torch.no_grad():
    progress_bar = tqdm(range(len(test_dataset)))
    val_loss = 0

    for batch_idx in progress_bar:
      progress_bar.set_description('Val loss: {}'.format(val_loss / (batch_idx + 1)))
      example = next(val_examples)[1]
      embeddings = transformation_fn(example)

      val_loss += val_step(encoder, decoder, loss_fn, embeddings, USE_CUDA)[0]
       

    if last_val_loss is None or val_loss < last_val_loss:
      last_val_loss = val_loss

      torch.save(encoder.state_dict(), 'checkpoints/encoder_{}.pt'.format(epoch))
      torch.save(decoder.state_dict(), 'checkpoints/decoder_{}.pt'.format(epoch))


# %%


