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

k_means = KMeans(n_clusters=5, init='k-means++', max_iter=100)
t0 = time()

k_means.fit(X)
yk_means = k_means.predict(X)

centroids = k_means.cluster_centers_

xtr, xt, ytr, yt = train_test_split(X, yk_means, test_size=0.1, random_state=42, stratify=yk_means)

DTreeCl= DecisionTreeClassifier(max_depth=5, random_state=42).fit(xtr, ytr)

pred_y = DTreeCl.predict(xt)

acc = accuracy_score(yt, pred_y)
prec = precision_score(yt, pred_y, average='weighted', zero_division=1)
reca = recall_score(yt, pred_y, average='weighted')
f1 = f1_score(yt, pred_y, average='weighted')

print(f"acc:{acc}, prec:{prec}, reca:{reca}, f1:{f1}")

classes = [f"Cluster ({i})" for i in range(5)]

dt_figure = plt.figure(figsize=(30,10))
_ = plot_tree(DTreeCl, 
              feature_names=names,
              class_names=classes,
              filled=True,
              fontsize=5)

plt.show()

C = k_means.cluster_centers_.argsort()[:, ::-1]
for i in range(5):
    print("Cluster %d:" %i, end='')
    for ix in C[i, :10]:
        print(" %s" %names[ix], end='')
    print()



