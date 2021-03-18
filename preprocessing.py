import pandas as pd
import torch
import numpy as np
from langdetect import detect
import nltk
import re

def language_detector(string):
    try:
        res = detect(string)
    except:
        res = "undetectable"
    return res
        
def makeDicts(word_set):
    word_to_int = dict((c, i) for i, c in enumerate(word_set))
    int_to_word = dict((i, c) for i, c in enumerate(word_set))
    return(word_to_int, int_to_word)

"""
1.Preprocess
Stuff to do for pre-processing:
Take out rows with NaN in "genre" or "lyrics" columns
Replace new line, exlamation, and question mark with itself followed by space
Take out songs with some variation of "There are no lyrics here", "Lyrics for this song have yet to be released. Please check back once the song has been released."
Take out songs with less than x words. Find good x value.
Delete songs with "RAP GENIUS" in lyrics
Remove songs with
Convert letters to lowercase
Replace blank lines with blank string
Remove special characters
2.Tokenize
3.Create word embeddings
"""

def lowerAndReplaceChars(text):
    stopChars = [',','(',')','.','-','[',']','"']
    processedText = text.lower()
    for char in stopChars:
        processedText = processedText.replace(char,'')
    return processedText


def clean_up(genius_df, langdetect):

    #Remove songs with too few words, and nans. As of now keep songs with > 10 words
    genius_df.dropna(subset=['genre', 'lyrics'], inplace=True)
    genius_df['word_num'] = genius_df['lyrics'].str.split().str.len()
    genius_df.sort_values(by = "word_num").head(100)
    genius_df['word_num'].astype('int32')
    genius_df = genius_df[genius_df.word_num != 1]
    genius_df = genius_df[~genius_df['lyrics'].str.contains("RAP GENIUS")] # these are all screenplays
    genius_df = genius_df[genius_df['word_num'] != 18] #Change this eventually to hardcode unwanted lyrics
    genius_df = genius_df[genius_df['word_num'] > 10]

    #Drop rows with null after processing
    genius_df = genius_df[genius_df['lyrics'].notnull()]

    # Drop indices where lyrics is not a string
    bad_indices = []
    for index, value in genius_df['lyrics'].items():
        if type(value) != str:
            bad_indices.append(index)
    genius_df = genius_df.drop(bad_indices)

    #Only keep English lyrics
    if langdetect:
        genius_df['language'] = genius_df['lyrics'].apply(language_detector)
        genius_df = genius_df[genius_df['language'] == "en"]

    return genius_df

    
def get_tokenized_lyrics(cleaned_df):
    #Make lower, delete special chars
    cleaned_df['lyrics'] = cleaned_df['lyrics'].astype(str)
    cleaned_df['lyrics'] = cleaned_df['lyrics'].apply(lowerAndReplaceChars)
    
    all_lyrics = cleaned_df['lyrics'].str.cat(sep='\n')
    all_lyrics = str.replace(all_lyrics, '\n', ' \n ')
    all_lyrics = str.replace(all_lyrics, '?', ' ! ')
    all_lyrics = str.replace(all_lyrics, '!', ' ? ')
    tokens = re.findall(r'\S+|\n', all_lyrics)
    return tokens

def tokenize(df, langdetect=True):
    cleaned_df = clean_up(df, langdetect=langdetect)
    return get_tokenized_lyrics(cleaned_df)

def getVocabulary(tokens):
    return sorted(set(tokens))