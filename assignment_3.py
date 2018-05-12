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
import math
from PIL import Image
from pathlib import Path
from skimage.data import imread
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from nltk.text import TextCollection
from itertools import chain
import operator
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

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

def filter_headlines(headlines):
    print ("Filtering all headlines...")
    filtered_headlines = list();
    
    for i in range (1, len(headlines)):
        headline_split = headlines.iloc[i, 0].split()
        
        #Remove numbers
        head_wo_numbers = [word for word in headline_split if not word.isdigit()]
        
        #Remove stopwords
        head_wo_stopwords = [word for word in head_wo_numbers if word not in stopwords.words('english')]
        
        #Find stem words
        porter = PorterStemmer()
        stemmed_words = [porter.stem(word) for word in head_wo_stopwords]
        
        #Append current headline to filtered headlines array
        filtered_headlines.append(" ".join(stemmed_words))
        
    return filtered_headlines

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

#OK - Here we do not tokenize.
def freq(ngram, headline):
    return headline.count(ngram)

#OK
def word_count(headline):
    return len(headline.split())

#OK
def tf(ngram, headline):
    #return (freq(ngram, headline) / float(word_count(headline)))
    return freq(ngram, headline)

#OK
def num_headlines_containing(ngram, headlines):
    count = 0
    for headline in headlines:
        if freq(ngram, headline) > 0:
            count += 1
    return 1 + count

#OK
def idf(ngram, headlines):
    return math.log(len(headlines) /
            (1.0 + float(num_headlines_containing(ngram, headlines))))
    
#OK
#headline and headlines are NOT tokenized yet
def tf_idf(ngram, headline, headlines):
    return (tf(ngram, headline) * idf(ngram, headlines))

def extract_ngrams(filtered_headlines):
    print("Extracting char-4-grams...")
    
    allgrams = list()

    #Extracting char-4-grams for each headline    
    for i in range (len(filtered_headlines)):
        curr_grams = ngrams(filtered_headlines[i], 4)
        curr_grams_list = list(curr_grams)
        for j in range (len(curr_grams_list)):
            allgrams.append ("".join(curr_grams_list[j]))
    
    #Calculating all frequencies to obtain char-4-grams without repetition
    fdist = nltk.FreqDist(allgrams)
    
    #Transforming FreqDist hash in a list of tuples ordered by number of occurrences
    sorted_fdist = sorted(fdist.items(), key = operator.itemgetter(1))
    
    sorted_fdist = sorted_fdist[90000:]
    
    return sorted_fdist, allgrams

#Extract matrix of tf values 
def extract_tf_matrix(filtered_headlines, sorted_fdist):
    tf_matrix = np.zeros([len(filtered_headlines), len(sorted_fdist)])
    print ("Extracting tf matrix... ")
    
    for i in range (0, len(filtered_headlines)): #For all headlines. STARTS IN 0 IF LIST, 1 IF PANDAS DATAFRAME
        #print ("Processing headline #" + str(i))
        feat_vec = [0] * len(sorted_fdist) #New feature array with the size of the hash

        for j in range (len(sorted_fdist)): #For each entry in the n-gram frequency list
            curr_ngram = sorted_fdist[j][0] #Get current n-gram
            
            #Frequency of every ngram in headline[i]
            feat_vec[j] = tf(curr_ngram, filtered_headlines[i])
        
        tf_matrix[i,:] = feat_vec
        
    return tf_matrix

#Extract matrix of idf values
def extract_idf_matrix(tf_matrix, filtered_headlines, sorted_fdist):
    
    size = tf_matrix.shape[1]
    idf_matrix = np.zeros([size, size])
    
    #For every ngram, calculate its idf
    for i in range (size):
        curr_idf = idf(sorted_fdist[i][0], filtered_headlines)
        idf_matrix[i,i] = curr_idf
        
    return idf_matrix

#Multiply tf_matrix by idf_matrix resulting in tfidf_matrix
def extract_tfidf_matrix(tf_matrix, idf_matrix):
    tfidf_matrix = np.dot(tf_matrix, idf_matrix)
    
    return tfidf_matrix
    
def normalize_matrix(matrix):
    norm_matrix = normalize(matrix)
    
    return norm_matrix


def build_feature_set(stem_heads, filtered_sorted_fdist):
    feature_matrix = np.zeros([len(stem_heads), len(filtered_sorted_fdist)])
    
    #len(stem_heads)
    for i in range (0, len(stem_heads)): #For all headlines. STARTS IN 0 IF LIST, 1 IF PANDAS DATAFRAME
        #print ("Processing headline #" + str(i))
        feat_vec = [0] * len(filtered_sorted_fdist) #New feature array with the size of the hash
        
        for j in range (len(filtered_sorted_fdist)): #For each entry in the n-gram frequency list
            curr_ngram = filtered_sorted_fdist[j][0] #Get current n-gram
            
            #if curr_ngram in stem_heads.iloc[i][1]: #Check if headline contains n-gram
            if curr_ngram in stem_heads[i]:
                feat_vec[j] = 1
        
        feature_matrix[i,:] = feat_vec
        
    
    return feature_matrix

def extract_filtered_headlines_by_year (dates, headlines, year):
    print("Extracting headlines for year " + year + "...")
    
    year_headlines = list();
    
    for i in range(1, len(dates)):
        currYear = str(dates.iloc[i,0])[:4] #Get headline year
        if currYear[:4] == year:
            year_headlines.append(headlines[i-1])
            
    return year_headlines

#Running PCA gives the same result as not running, so there's no need to use it
def run_PCA (feature_matrix):
    pca = PCA(svd_solver = 'auto')
    pca.fit(feature_matrix)
    X = pca.transform(feature_matrix)
    
    return X

def main():
    year = '2003'
    
    dates, headlines = read_input()
    filtered_headlines = filter_headlines(headlines) #list of processed headlines
    sorted_fdist, allgrams = extract_ngrams(filtered_headlines) #sorted ngrams by frequency
    
    #year_headlines = extract_filtered_headlines_by_year (dates, filtered_headlines, year) #Get only headlines of defined year

    #First, execute for all headlines
    tf_matrix = extract_tf_matrix(filtered_headlines, sorted_fdist) #Calculate frequency matrix
    idf_matrix = extract_idf_matrix(tf_matrix, filtered_headlines, sorted_fdist); #Calculate idf matrix
    tfidf_matrix = extract_tfidf_matrix(tf_matrix, idf_matrix) #Calculate tfidf matrix
    norm_tfidf_matrix = normalize_matrix(tfidf_matrix) #Normalize tfidf matrix

    k = 19

    mbk = mbkmeans(k, norm_tfidf_matrix)
    #run_mini_batch_kmeans(year, norm_tfidf_matrix, 1, 21) #Run KMeans

    return mbk


def mbkmeans (k, feature_matrix):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=100, n_init=10, max_no_improvement=10, verbose=0)
    mbk.fit(feature_matrix)
    cost = mbk.inertia_
    
    return mbk

def run_mini_batch_kmeans(year, feature_matrix, k_start, k_end):
    for i in range (k_start, k_end):
        print("cost for year: " + str(year) + " and k= " + str(i) + " = " + str(mbkmeans(i, feature_matrix)))

def kmeans(k, feature_matrix):
    kmeans = KMeans(n_clusters = k, random_state = 0).fit(feature_matrix)
    cost = kmeans.inertia_
    
    return cost

def run_kmeans(feature_matrix, k_start, k_end):
    for i in range (k_start, k_end):
        print("cost k=" + str(i) + " = " + str(kmeans(i, feature_matrix)))
        
#Find 1-grams of all headlines from cluster k in order to be easily analyzable
def analyze_result(k, mbk, headlines):
    cluster_map = pd.DataFrame()
    cluster_map['cluster'] = mbk.labels_ 
    cluster_map['headline'] = headlines
    
    curr_map = cluster_map.loc[cluster_map['cluster'] == k] #Separate by cluster
    curr_headline_list = list(curr_map['headline']) #Get only headlines for current cluster
    #here, find tokens
    
    all_ngrams = list();
    for headline in curr_headline_list:
        onegrams = ngrams(headline.split(), 1)
        all_ngrams.append(onegrams)
        
    #Here we have all 1-grams for all headlines of current cluster
    #Calculating all frequencies
    fdist = nltk.FreqDist(all_ngrams)
    
    #Transforming FreqDist hash in a list of tuples ordered by number of occurrences
    sorted_fdist = sorted(fdist.items(), key = operator.itemgetter(1))
    
    return sorted_fdist
    
        
    
        
    