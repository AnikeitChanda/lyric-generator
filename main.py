from preprocessing import tokenize, getVocabulary, makeDicts
from lstm import make_dataloaders, train_model, Simple_LSTM, test
import pandas as pd
import torch

def main():
    genius_df = pd.read_csv('raw-genius-data.csv')
    genius_df = genius_df.sample(10, random_state = 69)
    tokens = tokenize(genius_df, langdetect=False) # list of tokens representing song lyrics tokens
    vocab = getVocabulary(tokens)
    word2idx, idx2word = makeDicts(vocab)
    seq_length, step = 16, 1 #up to us to choose
    # Train
    # dataloader = make_dataloaders(tokens, seq_length, step, word2idx, 32)
    # model = train_model(dataloader, 10, len(vocab))
    # torch.save(model.state_dict(), 'rap_checkpoint.pth')
    # Test
    trained_model = Simple_LSTM(len(vocab),256,50)
    trained_model.load_state_dict(torch.load('rap_checkpoint.pth'))
    test(trained_model, idx2word, word2idx, seq_length=seq_length)

if __name__ == '__main__':
    main()