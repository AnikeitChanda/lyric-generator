from preprocessing import tokenize, getVocabulary, makeDicts, get_better_embeddings, sentence2seed
from lstm import make_dataloaders, train_model, Simple_LSTM, test, generate
import pandas as pd
import torch
import argparse
from nltk.translate.bleu_score import sentence_bleu

def eval(model, word2idx, idx2word, df, windowSize, device):
	n_songs = list(df["lyrics"].sample(10, random_state=10))
	scores = []
	for song in n_songs:
		tokens = sentence2seed(song)
		seed = tokens[:windowSize]
		generatedLyrics = generate(model, idx2word, word2idx, windowSize, seed, device)[windowSize:] # size 400
		trueLyrics = tokens[windowSize:]
		trueLyrics = trueLyrics[:400]
		score = sentence_bleu([generatedLyrics], trueLyrics)
		scores.append(score)
	print(f"Average Bleu Score: {sum(scores)/len(scores)}")
	
	

def main(seq_length, step, epochs, weights=None, outputPath=None, evalOnly=False):
    if torch.cuda.is_available():
      device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
      print("Running on the GPU")
    else:
      device = torch.device("cpu")
      print("Running on the CPU")
    genius_df = pd.read_csv('cleaned_lyrics_new.csv')
    tokens = tokenize(genius_df, langdetect=False)
    embeddings, word2idx, idx2word = get_better_embeddings(tokens)
#     genius_df = genius_df.loc[genius_df['genre'] == 'pop']
    genius_df = genius_df.sample(20)
    tokens = tokenize(genius_df, langdetect=False) # list of tokens representing song lyrics tokens
    vocab = getVocabulary(tokens)

#     trained_model = Simple_LSTM(len(idx2word),256,embeddings).to(device)
# #     trained_model.load_state_dict(torch.load('16_1_5.pth'))
# #     seedSentence = " ".join(['i', 'looked', 'now', 'doo', 'quick', 'get', 'call', 'if', 'just', 'when', 'oh', 'i', 'what', '\n', 'when', "couldn't"])
# #     test(trained_model, idx2word, word2idx, seq_length, seedSentence, device)

    if outputPath is None:
      outputPath = f"{seq_length}_{step}_{epochs}.pth"
    # Train
    if not evalOnly:
      dataloader = make_dataloaders(tokens, seq_length, step, word2idx, 256)
      model = train_model(dataloader, epochs, len(idx2word), embeddings, device, weights=weights)
      torch.save(model.state_dict(), outputPath)
    # Test
    trained_model = Simple_LSTM(len(idx2word),256,embeddings).to(device)
    if not evalOnly:
      trained_model.load_state_dict(torch.load(outputPath))
    else:
      if weigths is None:
        print("Provide weights using --weights")
        return
      trained_model.load_state_dict(torch.load(weights))

    #test(trained_model, idx2word, word2idx, seedSentence, seq_length, device)
    eval(trained_model, word2idx, idx2word, genius_df, seq_length, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_epochs = 1
    default_sequence_length = 16
    default_step_length = 1
    parser.add_argument("-e", "--epochs", default=default_epochs, type=int)
    parser.add_argument("-seq", "--sequence_length", default=default_sequence_length, type=int)
    parser.add_argument("--step", default=default_step_length, type=int)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output_file", default=None, type=str)
    args = parser.parse_args()
    main(args.sequence_length, args.step, args.epochs, weights=args.weights, outputPath=args.output_file, evalOnly=args.eval)