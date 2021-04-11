from preprocessing import tokenize, getVocabulary, makeDicts, get_better_embeddings
from lstm import make_dataloaders, train_model, Simple_LSTM, test
import pandas as pd
import torch

def main():
    if torch.cuda.is_available():
      device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
      print("Running on the GPU")
    else:
      device = torch.device("cpu")
      print("Running on the CPU")
    genius_df = pd.read_csv('cleaned_lyrics_new.csv')
    tokens = tokenize(genius_df, langdetect=False)
    embeddings, word2idx, idx2word = get_better_embeddings(tokens)
    #genius_df = genius_df.sample(10, random_state = 69)
    genius_df = genius_df.loc[genius_df['genre'] == 'pop']
    tokens = tokenize(genius_df, langdetect=False) # list of tokens representing song lyrics tokens
    # vocab = getVocabulary(tokens)
    # word2idx, idx2word = makeDicts(vocab)
    seq_length, step = 16, 2 #up to us to choose
    # Train
    dataloader = make_dataloaders(tokens, seq_length, step, word2idx, 128)
    model = train_model(dataloader, 10, len(idx2word), embeddings, device)
    torch.save(model.state_dict(), 'rap_checkpoint.pth')
    # Test
    trained_model = Simple_LSTM(len(idx2word),256,embeddings).to(device)
    trained_model.load_state_dict(torch.load('rap_checkpoint.pth'))
    test(trained_model, idx2word, word2idx, seq_length=seq_length, device=device)

if __name__ == '__main__':
    main()