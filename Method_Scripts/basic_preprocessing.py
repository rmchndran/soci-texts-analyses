import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import *
from pdftotext import pdftotext
from detect_encoding import detect_encoding

def preprocess_I(path):
    output, filename = pdftotext(path)
    fp = f"{filename}.txt"
    type = detect_encoding(fp)
    strip, n = preprocess_fun(fp, type)
    input, n = writeTo(strip, n, type)
    return remove_stopwords(n, type), type

def mle(chunked_words, df, plot):
    rejoined = ' '.join(chunked_words)
    resplit = rejoined.split()
    per_word_freq = pd.Series(resplit).value_counts()

    wp_mle = per_word_freq / len(resplit)
    wp_mle_sorted = wp_mle.sort_values(ascending=False)

    top_words_by_mle = wp_mle_sorted.head(20).index
    
    mle_freq_df = pd.DataFrame(0, index=np.arange(len(df)), columns=top_words_by_mle)

    for i, tengram in enumerate(df['10-chunk']):
        tengram_split = tengram.split()
        mle_freq_df.loc[i, tengram_split] = mle_freq_df.loc[i, tengram_split] + 1

    
    plot[1].set_title('Top 10 [MLE]')
    for i in mle_freq_df:
        plot[1].plot(mle_freq_df, label=i)

    plot[1].set_ylabel('Frequency')
    plot[1].set_xlabel('Chunks | 10-gram')
    plot[1].legend()

    plt.show()

text_file, encoding = preprocess_I("/Users/andreas/Desktop/DCLH_Research/disregard/comte/presentation_of_self.pdf")

with open(text_file, 'r') as rf:
    file = rf.read()
    ngram = file.split()
    word_groups = [' '.join(ngram[word:word+10]) for word in range(0, len(ngram), 10)]
    word_group_df = pd.DataFrame(word_groups, columns=['10-chunk'])

figure, plot = plt.subplots(2,1, figsize=(14,12))

mle(word_groups, word_group_df, plot)