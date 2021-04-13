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
from preprocessing import sentence2seed
# nltk.download('punkt')

# Revisit if we run into memory issues in colab
class SequencesDataset(Dataset):
    def __init__(self, DP_text, seq_length, step, word2idx):
        self.DP_text = DP_text
        self.seq_length = seq_length
        self.step = step
        self.word2idx = word2idx

    def __len__(self):
        return len(range(0, len(self.DP_text) - self.seq_length, self.step))

    def __getitem__(self, idx):
        sentence, word = self.DP_text[idx * self.step: idx*self.step + self.seq_length], self.DP_text[idx*self.step + self.seq_length]
        idx_word = self.word2idx[word]
        idx_sentence = [self.word2idx[word_] for word_ in sentence]
        return np.array(idx_sentence), idx_word

def make_dataloaders(DP_text, seq_length, step, word2idx, b_size):
    train_dataset = SequencesDataset(DP_text, seq_length, step, word2idx)
    return torch.utils.data.DataLoader(train_dataset, batch_size = b_size)
    


class Simple_LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, glove_emb_weights, dropout = 0.2):
        super(Simple_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(glove_emb_weights, freeze = False)
        self.lstm = nn.LSTM(glove_emb_weights.shape[1], hidden_dim,dropout = dropout,num_layers = 2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, seq_in):
        # for LSTM, input should be (Sequnce_length,batchsize,hidden_layer), so we need to transpose the input
        embedded = self.embeddings(seq_in.t()) 
        lstm_out, _ = self.lstm(embedded)
        # Only need to keep the last word
        ht=lstm_out[-1] 
        out = self.fc(ht)
        return out

def train_model(dataloader, epoch_count, vocab_size, glove_emb, device, weights=None):
    #Change embedding dim and hidden dim
    model = Simple_LSTM(vocab_size,256,glove_emb).to(device)
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
    
    avg_losses_f = []

    for epoch in range(epoch_count):
        print("Epoch: ", epoch)
        start_time = time.time()
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        avg_loss = 0.
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
        
            optimizer.step()
            avg_loss+= loss.item()/len(dataloader)
            if i % 1000 == 0:
                print(avg_loss)

        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, epoch_count, avg_loss, elapsed_time))
    
        avg_losses_f.append(avg_loss)


    print('All \t loss={:.4f} \t '.format(np.average(avg_losses_f)))

    plt.plot(avg_losses_f)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.show()
    
    
    return(model)



def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def test(model, idx2word, word2idx, seq_length, sentence, device):
    model.eval()
    seed = sentence2seed(sentence)
    generated = []
    original = seed
    window = seed[-seq_length:]
    variance = 0.9
    for i in range(400):
        x = np.zeros((1, seq_length))   
        for t, word in enumerate(window):
            x[0, t] = word2idx[word] # Change the sentence to index vector shape (1,50)
        
        x_in = Variable(torch.LongTensor(x))
        x_in = x_in.to(device)
        pred = model(x_in)
        pred = np.array(F.softmax(pred, dim=1).data[0].cpu())
        next_index = sample(pred, variance)
        next_word = idx2word[next_index] # index to word

        generated = generated + [next_word]
        window = window[1:] + [next_word] # Update Window for next char predict
    
    print(" ".join(original + generated))

    


