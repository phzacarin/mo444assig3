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
from itertools import chain

def read_input():
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Assignments/3/news_headlines.csv')
    file_contents = pd.read_csv(file_path, header=None)
    dates = file_contents.iloc[:,0:1]
    headlines = file_contents.iloc[:,1:2]
    
    return dates, headlines
    
def headlines_2_text(headlines):
    all_headlines = ""
    for i in range (1, len(headlines)):
        all_headlines = all_headlines + headlines.iloc[i, 0]
        all_headlines = all_headlines + " "
        
    return all_headlines
    
def extract_ngrams(content):
    allgrams = nltk.everygrams(content.split(), 2, 4)
    fdist = nltk.FreqDist(allgrams)
    return fdist, allgrams
