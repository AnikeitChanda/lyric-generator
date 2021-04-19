import os
from preprocessing import tokenize_lm
import pandas as pd
import nltk
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from collections import Counter
import random



def load_data(train_path, genre):
    genius_df = pd.read_csv(train_path)
    tokens = tokenize_lm(genius_df, genre, langdetect=False)
    return tokens


class LM():
    def __init__(self, tokens, lam):
        self.lam = lam
        self.tokens = tokens
        self.vocab = set(self.tokens)


    def make_counts(self):
        unigrams = nltk.ngrams(self.tokens, 1)
        bigrams = nltk.ngrams(self.tokens, 2)
        trigrams = nltk.ngrams(self.tokens, 3)
        self.unigram_counts = dict(nltk.FreqDist(unigrams))
        self.bigram_counts = dict(nltk.FreqDist(bigrams))
        self.trigram_counts = dict(nltk.FreqDist(trigrams))


    def create_trigram_model(self):
        self.make_counts()
        model = {}
        for k, v in self.trigram_counts.items():
            w1, w2, w3 = k
            if (w1, w2) not in model.keys():
                model[(w1, w2)] = {}
                model[(w1, w2)][w3] = v + self.lam
            else:
                model[(w1, w2)][w3] = v + self.lam

        for k, v in model.items():
            tmp_sum = sum(v.values())
            tmp_sum += self.lam * len(self.vocab)
            w3_list = v.keys()
            for w3 in w3_list:
                model[k][w3] /= tmp_sum
        return model


    def create_bigram_model(self):
        model = {}
        for k, v in self.bigram_counts.items():
            w1, w2 = k
            if w1 not in model.keys():
                model[w1] = {}
                model[w1][w2] = v + self.lam
            else:
                model[w1][w2] = v + self.lam

        for k, v in model.items():
            tmp_sum = sum(v.values())
            tmp_sum += self.lam * len(self.vocab)
            w2_list = v.keys()
            for w2 in w2_list:
                model[k][w2] /= tmp_sum

        return model


    def create_unigram_model(self):
        model = self.unigram_counts
        tmp_sum = sum(model.values()) + (self.lam * len(self.vocab))
        for k in model.keys():
            model[k] += self.lam
            model[k] /= tmp_sum
        return model


    def create_interpolation_model(self):
        tri_lam = 0.5
        bi_lam = 0.4
        uni_lam = 0.1
        trigram_model = self.create_trigram_model()
        bigram_model = self.create_bigram_model()
        unigram_model = self.create_unigram_model()
        for k, v in trigram_model.items():
            inner_ks = v.keys()
            for inner_k in inner_ks:
                trigram_model[k][inner_k] *= tri_lam
        for k, v in bigram_model.items():
            inner_ks = v.keys()
            for inner_k in inner_ks:
                bigram_model[k][inner_k] *= bi_lam
        for k, v in unigram_model.items():
            unigram_model[k]*= uni_lam
        return unigram_model, bigram_model, trigram_model


    def generate_lyrics(self, start):
        self.make_counts()
        unigram_model, bigram_model, trigram_model = self.create_interpolation_model()
        sentence = start
        model_inp = start.split()
        count = 0
        new_word = ""
        while count < 40 and new_word != '</s>':
            if (model_inp[len(model_inp)-2], model_inp[len(model_inp)-1]) in trigram_model.keys():
                trigram_options = trigram_model[(model_inp[len(model_inp)-2], model_inp[len(model_inp)-1])]
            else:
                trigram_options = {}
            if model_inp[len(model_inp)-1] in bigram_model.keys():
                bigram_options  = bigram_model[model_inp[len(model_inp)-1]]
            else:
                bigram_options = {}

            options_dic = trigram_options.copy()
            for k, v in bigram_options.items():
                if k in options_dic.keys():
                    options_dic[k] += v
                else:
                    options_dic[k] = v

            top_k = Counter(unigram_model)
            unigram_options = top_k.most_common(40)
            for k, v in unigram_options:
                if k[0] in options_dic.keys():
                    options_dic[k[0]] += v
                else:
                    options_dic[k[0]] = v
            sorted_options = []
            for k, v in options_dic.items():
                sorted_options.append((k, v))
            sorted_options = sorted(sorted_options, key=lambda option: option[1], reverse=True)

            if len(sorted_options) >= 10:
                rand_idx = random.randint(0, 10)
            else:
                rand_idx = random.randint(0, len(sorted_options) - 1)
            new_word = sorted_options[rand_idx][0]
            sentence = sentence + " " + new_word
            model_inp.append(new_word)
            count += 1
        return sentence




def main():
    data_path = 'cleaned_lyrics_new.csv'
    train_tokens = load_data(data_path, 'rock')
    lm = LM(train_tokens, 0.1)
    out = lm.generate_lyrics('on god i love the pussy, on god i love the blow')
    print(out)



if __name__ == '__main__':
    main()

