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

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

def getKeywords(str, numWords):
    r = Rake(min_length=1, max_length=1) # Only get single word phrases
    r.extract_keywords_from_sentences([str])
    return r.get_ranked_phrases()[:numWords]

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
        embedded = self.embeddings(keyword_indices.T) # embed keyword
        encoder_hidden = self.encoder(embedded)
        batchsize = keyword_indices.shape[0]
        outputs = torch.zeros(self.seq_length + 1, batchsize, self.vocab_size).to(device)
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
    def __init__(self, processedData):
        self.processedData = processedData
    
    def __len__(self):
        return len(self.processedData)
    
    def __getitem__(self, idx):
    	return self.processedData[idx]

def train(dataloader, model, num_epochs, criterion, optimizer, fill_value):
    model.train()
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = X.shape[0]
            X = X.to(device)
            y = y.to(device)
            startIndices = torch.full((1, batch_size), fill_value).to(device)
            out_seq = model(X, startIndices, y)
            out_dim = out_seq.shape[-1]
            out_seq = out_seq[1:].view(-1, out_dim)
            y = y.T[1:].reshape(-1)
            loss = criterion(out_seq, y)
            loss.backward()
            # need to add gradient clipping here or no??
            optimizer.step()
        print(f"Epoch {epoch} Complete")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def processLyrics(df, word2idx, seqLen, numKeywords):
    data = []
    for lyric in df.array:
        tokens = sentence2seed(lyric)
        keywords = getKeywords(" ".join(tokens), numKeywords)
        if len(keywords) != numKeywords:
        	continue
        seed = ["<start>"] + tokens[:seqLen]
        if len(seed) != seqLen + 1:
            continue
        try:
        	keywordIndices = [word2idx[word] for word in keywords]
        	seedIndices = [word2idx[word] for word in seed]
        	data.append((torch.tensor(keywordIndices), torch.tensor(seedIndices)))
        except KeyError:
        	pass
    return data

def generate(model, keywords, word2idx, idx2word):
    model.eval()
    generated = []
    keywordIndices = [word2idx[word] for word in keywords]
    keywordTensor = torch.tensor(keywordIndices).view(1, -1).to(device)
    startIndices = torch.full((1, 1), word2idx["<start>"]).to(device)
    out_seq = model(keywordTensor, startIndices, None, teacher_force_ratio=0.0) # torch.Size([17, 1, 113432])
    out_seq = out_seq[1:] # torch.Size([16, 1, 113432])
    out_seq = out_seq.squeeze(1).cpu() # torch.Size([16, 113432])
    for rowIndex in range(out_seq.shape[0]):
        row = out_seq[rowIndex]
        pred = np.array(F.softmax(row))
        bestIndex = sample(pred)
        generated.append(idx2word[bestIndex])
    print(generated)
    
   
def main():
    df = pd.read_csv("cleaned_lyrics_new.csv")
    NUMKEYWORDS = 5
    SEQLEN = 16
    tokens = tokenize(df, langdetect=False)
    embeddings, word2idx, idx2word = get_better_embeddings(tokens, True)
    df = df.sample(48193//3, random_state=100)
    processedData = processLyrics(df["lyrics"], word2idx, SEQLEN, NUMKEYWORDS)
    print(f"Processed Lyrics: {len(processedData)}")
    trainDataset = SeedDataSet(processedData)
    dataloader = DataLoader(trainDataset, batch_size=128)
    model = SeedGenerator(len(idx2word), 100, embeddings, SEQLEN).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train(dataloader, model, 2, criterion, optimizer, word2idx["<start>"])
    keywords = ["young", "love", "fun", "happy", "world"]
    assert len(keywords) == NUMKEYWORDS
    with torch.no_grad():
        generate(model, keywords, word2idx, idx2word)
        print(idx2word)

if __name__ == "__main__":
    main()