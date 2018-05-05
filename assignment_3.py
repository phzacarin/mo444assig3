#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:04:51 2018

@author: Pedro
"""
import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from skimage.data import imread
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from itertools import chain
import operator

def read_input():
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Assignments/3/news_headlines.csv')
    file_contents = pd.read_csv(file_path, header=None)
    dates = file_contents.iloc[:,0:1]
    headlines = file_contents.iloc[:,1:2]
    
    return dates, headlines
    
def headlines_2_text(headlines):
    all_headlines = ""
    for i in range (1, 10000): """NUMERO ARBITRARIO PARA TESTES SOMENTE"""
        all_headlines = all_headlines + (headlines.iloc[i, 0]).lower()
        all_headlines = all_headlines + " "
    
    words_wo_numbers = [word for word in all_headlines.split() if not word.isdigit()];
    
    """Filter all stopwords"""
    filtered_words = [word for word in words_wo_numbers if word not in stopwords.words('english')]
    
    """Find all stem words"""
    porter = PorterStemmer()
    stemmed_words = [porter.stem(word) for word in filtered_words]
        
    """Concatenates filtered words back in a single string"""
    filtered_headlines = " ".join(stemmed_words)
    
    return all_headlines, filtered_headlines
    
def extract_ngrams(content):
    allgrams = nltk.everygrams(content.split(), 2, 4)
    fdist = nltk.FreqDist(allgrams)
    
    """Transforming FreqDist hash in a list of tuples ordered by number of occurrences"""
    sorted_fdist = sorted(fdist.items(), key = operator.itemgetter(1))
    
    """Removing unary frequencies"""
    filtered_sorted_fdist = list()
    
    for i in range (len(sorted_fdist)):
        if sorted_fdist[i][1] != 1:
            filtered_sorted_fdist.append(sorted_fdist[i])
    
    return filtered_sorted_fdist, allgrams


def build_feature_set(headlines, fdist, allgrams):
    ngrams_dict = {}
    counter = 0
    
    """Building a hashmap with all n-grams, with an iterative counter as key"""
    for ngram in allgrams:
        ngrams_dict[counter] = ngram
        counter = counter + 1
        
    
        
    
        
    