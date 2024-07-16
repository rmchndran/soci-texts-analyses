from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, f1_score
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from time import time

with open("genview(lem&tok).txt", encoding="utf-16-le") as f:
    file = f.read()
    file_split = file.split()
    
    word_matrix = [' '.join(file_split[i:i+5]) for i in range(0, len(file_split), 5)]


V  = TfidfVectorizer(stop_words='english')
X = V.fit_transform(word_matrix)
    
names = V.get_feature_names_out()

print(X.shape)

n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)
T = lda.transform(X)

def topics(model, fnames, n_top_words):
    for tid, t in enumerate(model.components_):
        print(f"Topic: {tid}")
        print(' '.join([fnames[i] for i in t.argsort()[:-n_top_words - 1:-1]]))

n_top_words = 8
topics(lda, names, n_top_words)