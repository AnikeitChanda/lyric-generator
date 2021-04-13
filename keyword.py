from rake_nltk import Rake
from preprocessing import sentence2seed
import pandas as pd
import torch
import numpy as np
from langdetect import detect
import nltk
import re
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from preprocessing import sentence2seed, tokenize, get_better_embeddings
import random
from torch import optim


def getKeywords(str):
    r = Rake(min_length=1, max_length=1) # Only get single word phrases
    r.extract_keywords_from_sentences([str])
    return r.get_ranked_phrases()[:5]

class SeedGeneratorEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SeedGeneratorEncoder, self).__init__()   
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    def forward(self, keywords):
        # keywords.shape = num_keywords, 32 , embedding_dim
        output, hidden = self.lstm(keywords)
        return hidden

class SeedGeneratorDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(SeedGeneratorDecoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        # input is embedded token, input.shape = 32, embedding_dim
        input = F.relu(input)
        output, hidden = self.lstm(input, hidden)
        prediction = self.fc(output.squeeze(0))

        return prediction, hidden

class SeedGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim, glove_emb_weights, seq_length, dropout = 0.2):
        super(SeedGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(glove_emb_weights, freeze = False)
        self.encoder = SeedGeneratorEncoder(glove_emb_weights.shape[1], hidden_dim)
        self.decoder = SeedGeneratorDecoder(glove_emb_weights.shape[1], hidden_dim, vocab_size)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def forward(self, keyword_indices, seq_indices, true_values, teacher_force_ratio = 0.5):
        # seq will be a single keyword
        embedded = self.embeddings(keyword_indices.t()) # embed keyword
        encoder_hidden = self.encoder(embedded)
        batchsize = keyword_indices.shape[0]
        outputs = torch.zeros(self.seq_length + 1, batchsize, self.vocab_size)
        # outputs[0] = seq_indices
        count = 0
        tokenEmbeddings = self.embeddings(seq_indices)
        while count < self.seq_length:
            start_token, encoder_hidden = self.decoder(tokenEmbeddings, encoder_hidden)
            bestWord = start_token.argmax(1)
            count += 1
            outputs[count] = start_token
            teacher_force = random.random() < teacher_force_ratio
            # bestWord = true_values[count] if teacher_force else bestWord
            tokenEmbeddings = self.embeddings(bestWord)
            tokenEmbeddings = tokenEmbeddings.unsqueeze(0)
        return outputs


class SeedDataSet(Dataset):
    def __init__(self, df, word2idx, seq_len):
        self.df = df["lyrics"]
        self.word2idx = word2idx
        self.seq_length = seq_len
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        lyrics = self.df.iloc[idx]
        tokens = sentence2seed(lyrics)

        keywords = getKeywords(" ".join(tokens))
        seed = ["<start>"] + tokens[:self.seq_length]
        keywordIndices = [self.word2idx[word] for word in keywords]
        seedIndices = [self.word2idx[word] for word in seed]
        return torch.tensor(keywordIndices), torch.tensor(seedIndices)

def train(dataloader, model, num_epochs, criterion, optimizer, fill_value):
    model.train()
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = X.shape[0]
            startIndices = torch.full((1, batch_size), fill_value)
            out_seq = model(X, startIndices, y)
            out_dim = out_seq.shape[-1]
            out_seq = out_seq[1:].view(-1, out_dim)
            y = y.T[1:].reshape(-1)
            loss = criterion(out_seq, y)
            loss.backward()
            # need to add gradient clipping here or no??
            optimizer.step()

   
def main():
    df = pd.read_csv("cleaned_lyrics_new.csv")
    tokens = tokenize(df, langdetect=False)
    embeddings, word2idx, idx2word = get_better_embeddings(tokens, True)
    df = df.sample(100, random_state=25)
    trainDataset = SeedDataSet(df, word2idx, 16)
    dataloader = DataLoader(trainDataset, batch_size=32)
    model = SeedGenerator(len(idx2word), 100, embeddings, 16)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train(dataloader, model, 1, criterion, optimizer, word2idx["<start>"])

if __name__ == "__main__":
    main()