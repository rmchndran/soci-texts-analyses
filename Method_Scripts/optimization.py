import pandas as pd
import numpy as np
from scipy.optimize import minimize

original_doc = ""

dwc = {} 
w1 = "word"
guess_w = "tried"
lt = 10
alpha = 2
sentences = {}
raw_doc_length = len(original_doc)
dl = 100
length_con = 8


# 1 - Static Probability Function 

def word_given_author(doc_word_count : int, word : str, doc_vocab : dict, dataframe=None):
    target_word_count = doc_vocab.get(word, 0)
    return (target_word_count) / (doc_word_count) 

# 2 - Exponential Probability Model

def word_fit(word, length_target, decay_scalar):
    b = abs(len(word) - length_target)
    
    return np.exp(-decay_scalar * b)

# 3 - Basic Normalizing Vector:

def normalize_likelihood(word, doc_length, doc_vocab, doc_word_count):
    W_g_D = (doc_vocab.get(word, 0) / doc_word_count) / (doc_length)
    return W_g_D

# (according to likelihood estimate warp in local vs global data)...

# 4 - Weighting context (an attempt):

def get_prev(word, possible_prev, ngram_counts=dict):
    word_given_prev = ngram_counts.get((possible_prev, word), 0)

    count_prev = sum(count for (previous, _), count in ngram_counts.items() if previous == possible_prev)

    return (word_given_prev / count_prev) if count_prev > 0 else 0

# Syntactic/Grammatical Constraints:

def decode_constraint(word):
    return sum(1 for char in word if char.isalpha()) - len(word)

# Construct the objective function:

def obj_f(parameters):
    w1, doc_word_count = parameters
    w_given_a = word_given_author(doc_word_count, w1, dwc)
    length_comp = word_fit(w1, lt, alpha)
    w_given_prev = get_prev(w1, guess_w, sentences)
    normalize_by_length = normalize_likelihood(w1, raw_doc_length, dwc, doc_word_count)

    return - (w_given_a * length_comp * w_given_prev * normalize_by_length)

guess_w = [w1, dl]

result = minimize(obj_f, guess_w, constraints=[{'type':'eq', 'fun':lambda x: decode_constraint(x[0])}])