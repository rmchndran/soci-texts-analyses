import re
import os 
import spacy
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from textblob import TextBlob
import chardet


testfile = "/Users/andreas/Desktop/DCLH_Research/soci_texts/comte/genviewofpositivism.txt"
nlp = spacy.load("en_core_web_sm")

def preprocess_fun(file, enc):
    if file.endswith('.txt'):
        n = os.path.basename(file)
        name = re.sub(r"(\.txt$|\d)", '', n)
        with open(file, encoding=enc) as input:
            raw = input.read()
            lower = raw.lower()
            rem_punc = re.sub(r"[.!?,:;/'&*-_#$()]|[\"']",'',lower)
            strip = rem_punc.strip()
      
        return strip, name

def writeTo(text_content, name, enc):
    with open(f'{name}(processed).txt','w',encoding=enc) as input:
        input.write(text_content)
        return input, name

def remove_stopwords(name, enc):
    with open(f"{name}(processed).txt", encoding=enc) as input:
        raw = input.read()
        doc = nlp(raw)
        stop_removed = " ".join([word.text for word in doc if not word.is_stop])
        # sparse = [word for word in stop_removed if stop_removed.count(word) < 10]  

    with open(f'{name}(spacy).txt', 'w', encoding=enc) as  output:
        output.write(stop_removed)
    
    return name


def stem_porter(name, enc):
    Porter = PorterStemmer()
    with open(f"{name}(spacy).txt",encoding=enc) as input:
        raw = input.read()
        for w in raw.split():
            raw.replace(w, Porter.stem(w))

    with open(f"{name}(stemmed).txt", 'w' , encoding=enc) as output:
        output.write(raw)
    
    return name

def lemmatize(name, enc):
    with open(f"{name}(stemmed).txt", encoding=enc) as input:
        raw = input.read()
        l = WordNetLemmatizer()
        raw_tkn = word_tokenize(raw)
        for w in raw_tkn:
            raw.replace(w, l.lemmatize(w))

    with open(f"{name}(lem&tok).txt",'w', encoding=enc) as output:
        output.write(raw)
        file_name = f"{name}(lem&tok).txt"
    return name, file_name


def pos_and_chunk(name, enc):
    with open(f"{name}(lem&tok).txt",encoding=enc) as input:
        raw = input.read()
        tags = TextBlob(raw)
        tags_result = tags.tags

        pattern = "NP: {<DT>?<JJ>*<NN>}"
        standard_chunker = nltk.RegexpParser(pattern)
        shallow_parse = standard_chunker.parse(tags.tags)
        print(shallow_parse)

    return tags_result

def colloc_bigram(tags_result):
    word_list = [words for words,tags in tags_result]

    bigram = BigramCollocationFinder.from_words(word_list)
    bigram_assoc = bigram.nbest(BigramAssocMeasures.likelihood_ratio, 10)

    return bigram_assoc

def colloc_trigram(tags_result):
    word_list = [words for words,tags in tags_result]

    bigram = TrigramCollocationFinder.from_words(word_list)
    bigram_assoc = bigram.nbest(TrigramAssocMeasures.likelihood_ratio, 10)

    return bigram_assoc

# strip, name = preprocess(testfile)

# input, name2 = writeTo(strip, name)

# remove_stopwords(name)

# stem_porter(name)

# lemmatize(name)