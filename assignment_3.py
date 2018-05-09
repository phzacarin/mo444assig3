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
from nltk.util import ngrams
from itertools import chain
import operator
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def read_input():
    print("Loading all data...")
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Assignments/3/news_headlines.csv')
    file_contents = pd.read_csv(file_path, header=None)
    dates = file_contents.iloc[:,0:1]
    headlines = file_contents.iloc[:,1:2]
    
    return dates, headlines        

def stem_headlines(headlines):
    print ("Extracting stem words for all headlines...")
    for i in range(len(headlines)):
        porter = PorterStemmer()
        stemmed_words = [porter.stem(word) for word in headlines.iloc[i][1].split()]
        headlines.iloc[i][1] = " ".join(stemmed_words)
        
    return headlines

def headlines_2_text(headlines):
    print("Converting all headlines to text cleaning it in the process...")
    all_headlines = ""
    for i in range (1, len(headlines)): #Concatenate all headlines into a text. 10000 is for test purposes only
        all_headlines = all_headlines + (headlines.iloc[i, 0]).lower()
        all_headlines = all_headlines + " "
    
    words_wo_numbers = [word for word in all_headlines.split() if not word.isdigit()];
    
    #Filter all stopwords
    filtered_words = [word for word in words_wo_numbers if word not in stopwords.words('english')]
    
    #Find all stem words
    porter = PorterStemmer()
    stemmed_words = [porter.stem(word) for word in filtered_words]
        
    #Concatenates filtered words back in a single string
    filtered_headlines = " ".join(stemmed_words)
    
    return filtered_headlines
    
def extract_ngrams(filtered_headlines):
    print("Extracting char-4-grams...")
    
    #Extracting char-level 4-grams
    allgrams = ngrams(filtered_headlines, 4)
    
    #Casting into a list
    allgrams_list = list(allgrams)
    #Concatenating chars into a string
    for i in range (len(allgrams_list)):
        allgrams_list[i] = "".join(allgrams_list[i])
    
    #allgrams = nltk.everygrams(filtered_headlines.split(), 2, 4)
    
    #Calculating all frequencies
    fdist = nltk.FreqDist(allgrams_list)
    
    #Transforming FreqDist hash in a list of tuples ordered by number of occurrences
    sorted_fdist = sorted(fdist.items(), key = operator.itemgetter(1))
    
    #Removing unary frequencies
    filtered_sorted_fdist = list()
    
    for i in range (len(sorted_fdist)):
        if sorted_fdist[i][1] > 600: #Only get frequencies > n
            filtered_sorted_fdist.append(sorted_fdist[i])
    
    return filtered_sorted_fdist, allgrams

def build_feature_set(stem_heads, filtered_sorted_fdist):
    feature_matrix = np.zeros([len(stem_heads), len(filtered_sorted_fdist)])
    
    #len(stem_heads)
    for i in range (0, len(stem_heads)): #For all headlines. STARTS IN 0 IF LIST, 1 IF PANDAS DATAFRAME
        print ("Processing headline #" + str(i))
        feat_vec = [0] * len(filtered_sorted_fdist) #New feature array with the size of the hash
        
        for j in range (len(filtered_sorted_fdist)): #For each entry in the n-gram frequency list
            curr_ngram = filtered_sorted_fdist[j][0] #Get current n-gram
            
            #if curr_ngram in stem_heads.iloc[i][1]: #Check if headline contains n-gram
            if curr_ngram in stem_heads[i]:
                feat_vec[j] = 1
        
        feature_matrix[i,:] = feat_vec
        
    
    return feature_matrix

def extract_stem_headlines_by_year (dates, stem_heads, year):
    print("Extracting headlines for year " + year + "...")
    
    year_headlines = list();
    
    for i in range(1, len(dates)):
        currYear = str(dates.iloc[i,0])[:4] #Get headline year
        if currYear[:4] == year:
            year_headlines.append(stem_heads.iloc[i, 0])
            
    return year_headlines

#Running PCA gives the same result as not running, so there's no need to use it
def run_PCA (feature_matrix):
    pca = PCA(svd_solver = 'auto')
    pca.fit(feature_matrix)
    X = pca.transform(feature_matrix)
    
    return X

def kmeans(k, feature_matrix):
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(feature_matrix)
    cost = kmeans.inertia_
    
    return cost

def main():
    year = '2005'
    
    dates, headlines = read_input()
    stem_heads = stem_headlines(headlines)
    filtered_headlines = headlines_2_text(headlines)
    filtered_sorted_fdist, allgrams = extract_ngrams(filtered_headlines)
    year_headlines = extract_stem_headlines_by_year (dates, stem_heads, year)
    
    feature_matrix = build_feature_set(year_headlines, filtered_sorted_fdist)
    
    run_kmeans(feature_matrix)
    #pool = Pool(processes=4)
    #feature_matrix = [pool.apply(build_feature_set, args=(year_headlines, filtered_sorted_fdist))]
    
    #with Pool(processes=4) as pool:
    #    pool.starmap(build_feature_set, [(stem_heads, filtered_sorted_fdist)])
    
    return feature_matrix


def run_kmeans(feature_matrix, k_start, k_end):
    for i in range (k_start, k_end):
        print("cost i=" + str(i) + " = " + str(kmeans(i, feature_matrix)))
    
    
        
    
        
    