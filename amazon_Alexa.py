import pandas as pd

data=pd.read_csv("C:\\Users\exam.SBS\Desktop\\amazon_alexa.tsv",sep="\t")

#Tokenization

from nltk.tokenize import sent_tokenize
str="Hey man how are you ? Is Alexa working great?"
str= str.lower()

ws=sent_tokenize(str)
print(ws)

#Stop words

import string
punct=string.punctuation #assigning string.punctuation in punct

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

def stopfunc(wt1):
    filtered1=[]
    for w in wt1:
        if w not in stop_words:
            if w not in punct:  #punct=string.punctuation
                filtered1.append(w)
    print("Tokenized:",wt1)
    print("Filtered:",filtered1)
    return filtered1

#Lemmatization
def lemm(filtered1):
    str3=filtered1
    from nltk.stem.wordnet import WordNetLemmatizer
    lem=WordNetLemmatizer()
    lem_words=[]
    for w in str3:
        lem_words.append(lem.lemmatize(w,'v'))
    lem_words
    return (lem_words)
lemmat=lemm(filtered1)
lemmat