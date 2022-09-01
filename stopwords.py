import imp
from multiprocessing.sharedctypes import Value
from typing import Type
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import coo_array
import scipy.sparse as sp
import numpy as np
import scipy

RANDOM_ROUNDS = 5

class StopwordFinder():
    def __init__(self, txt, tf_vectorizer):
        """
        Inputs: 
            txt [pandas.Series] of all documents with space-sep tokens
            tf_vectorizer [sklearn.feature_extraction.text.CountVectorizer] 
                that has been initialized, but not neccessarily pre-fit
        """
        if not isinstance(tf_vectorizer, CountVectorizer):
            raise TypeError('tf_vectorizer must be of type sklearn.feature_extraction.text.CountVectorizer')
        
        # vectorize data
        tf_vectorizer.fit(txt)
        self.features = tf_vectorizer.transform(txt)
        
        # token name vector
        self.vocabulary = tf_vectorizer.get_feature_names()

        # term frequency
        tf = self.features.sum(axis=0).A[0]

        # get entropy
        entropy = calc_term_entropy(self.features)
        
        # randomize terms and get entropy
        words = txt.str.split(' ').explode()
        null = calc_null_entropy(tf_vectorizer, words.values, words.index.values)
        
        # entropy 
        inform = entropy * (1 - normalize(null - entropy))

        # rank difference
        raw_dshift = (scipy.stats.rankdata(inform) - scipy.stats.rankdata(tf))
        downshift = raw_dshift / len(inform)

        # semantic entropy
        self.semantic_entropy = inform + downshift * np.log(len(inform))

        # true means keep, false means remove
        self.stopword_mask = np.full(len(self.vocabulary), False, dtype=bool)
        self.threshold = np.inf

        # how likely it will show up in final LDA graph
        self.saliency = tf * (1 - normalize(raw_dshift))

    def add_percentile(self, p):
        """
        Select stopwords as terms above nth percentile lowest semantic value
        """
        self.threshold = np.percentile(self.semantic_entropy, p)
        self.stopword_mask = self.semantic_entropy >= self.threshold

    def add_total(self, n):
        """
        Select stopwords as least semantic value n terms
        """
        self.threshold = np.sort(self.semantic_entropy)[-n]
        self.stopword_mask = self.semantic_entropy >= self.threshold

    def add(self, stopwords):
        """
        Label a list of terms as stopwords
        """
        words = stopwords
        if not isinstance(words, list):
            words = list(words)
        vocab = list(self.vocabulary)
        for word in words:
            try:
                self.stopword_mask[vocab.index(word)] = True
            except ValueError:
                return

    def keep(self, keepwords):
        """
        Label a list of terms as not stopwords
        """
        words = list(keepwords)
        if not isinstance(words, list):
            words = list(words)
        vocab = list(self.vocabulary)
        for word in words:
            try:
                self.stopword_mask[vocab.index(word)] = False
            except ValueError:
                return

    @property
    def stopwords(self):
        """
        Return estimated least most salient terms in corpus for filtering
        """
        return np.ma.compressed(np.ma.masked_array(np.array(self.vocabulary), mask=~self.stopword_mask))

    @property
    def keywords(self):
        """
        Return estimated top most salient terms in corpus for filtering
        """
        stops = self.stopwords
        top_saliency = self.saliency.argsort()[::-1]
        top_saliency_terms = np.array(self.vocabulary)[top_saliency].tolist()
        return [t for t in top_saliency_terms if t not in stops]

    def filtered_features(self):
        """
        return sparse matrix of all features, without stopwords
        """
        return self.features @ sp.diags((~self.stopword_mask).astype(int))

def calc_term_entropy(tf_matrix):
    H = np.zeros(tf_matrix.shape[1])
    
    tf_matrix = coo_array(tf_matrix) # row col access
    tf_wc = tf_matrix.sum(axis=0)    # TF(w, C)
    
    for d, w, tf in zip(tf_matrix.row, tf_matrix.col, tf_matrix.data):            
        p_dw = tf / tf_wc[w]
        H[w] -= p_dw * np.log2(p_dw)
    
    return H

def calc_null_entropy(tf_vectorizer, words, doc_idx):
    if words.shape != doc_idx.shape:
        return None
    
    null_entropy = np.zeros(len(tf_vectorizer.get_feature_names()))
    
    random_rounds = RANDOM_ROUNDS
    for i in range(0, random_rounds):
        np.random.shuffle(words)
        null_txt = pd.Series(words, doc_idx).groupby(level=0).apply(' '.join).values
        null_entropy += calc_term_entropy(tf_vectorizer.transform(null_txt))

    return null_entropy / random_rounds
    
def normalize(arr):
    mx = arr.max()
    mn = arr.min()
    return (arr - mn) / (mx - mn)