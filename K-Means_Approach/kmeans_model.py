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

def kmeans(file, encoding, ngrams=int, plot=bool):
    with open(file, encoding=encoding) as input:
        file_raw = input.read()
        file = file_raw.split()

        word_matrix = [' '.join(file[i:i+ngrams]) for i in range(0, len(file), ngrams)]

    V = TfidfVectorizer()
    X = V.fit_transform(word_matrix)
    print(X.shape)

    feature_names = V.get_feature_names_out()

    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100)
    time0 = time()

    kmeans.fit(X)
    kmeans_predict = kmeans.predict(X)

    xtrain, xtest, ytrain, ytest = train_test_split(X, kmeans_predict, test_size=0.15, random_state=42, stratify=kmeans_predict)

    classifier = DecisionTreeClassifier(max_depth=5, random_state=42).fit(xtrain, ytrain)
    predict_y = classifier.predict(xtest)

    acc = accuracy_score(ytest, predict_y)
    prec = precision_score(ytest, predict_y, average='weighted', zero_division=1)
    reca = recall_score(ytest, predict_y, average='weighted')
    f1 = f1_score(ytest, predict_y, average='weighted')

    print(f"acc:{acc}, prec:{prec}, reca:{reca}, f1:{f1}")

    class_names = [f"Cluster {(i)}" for i in range(5)]

    C = kmeans.cluster_centers_.argsort()[:, ::-1]
    for i in range(5):
        print("Cluster %d:" %i, end='')
        for ix in C[i, :10]:
            print(" %s" %feature_names[ix], end='')
        print()

    if plot == True:
        
        dt_figure = plt.figure(figsize=(30,10))
        _ = plot_tree(classifier, 
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True,
                    fontsize=5)
        plt.show()