import nltk
from nltk.tokenize import word_tokenize
str1="India is exporting $10 million software services to USA"
str1= str1.lower()

## nltk.data.path  # to check the path of nltk
#import nltk
# nltk.download()

#Word Tokenization
wt=word_tokenize(str1)
print(wt)

#Sentence Tonkenization
from nltk.tokenize import sent_tokenize
str2="India is exporting $10 million software services to USA. Software is one of the most growing sector"
ws=sent_tokenize(str2)
print(ws)
ws[0]
ws[1]

#Frequency distribution
from nltk.probability import FreqDist
wt1=word_tokenize(str2)
fdist=FreqDist(wt1)
fdist.most_common(2)

import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()

#Part of Speech (POS)
pos=nltk.pos_tag(wt)
# nltk.help.upenn_tagset() #list of all tag i.e. there full forms of all tags(noun, adjective etc)
pos

#Stop words
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

filtered1=[]
for w in wt1:
    if w not in stop_words:
        filtered1.append(w)
print("Tokenized:",wt1)
print("Filtered:",filtered1)

#Lemmatization (it needs context)
str3="I am a runner running in the race as I love to run since I ran past years"
wt1=word_tokenize(str3)
from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()
lem_words=[]
for w in wt1:
    lem_words.append(lem.lemmatize(w,'v'))
lem_words

#Stemming
str3="connection connectivity connected connecting"  #['connect','connect','connect','connect']
str3="I am a runner running in the race as I love to run since I ran past years"
from nltk.stem import PorterStemmer
wt=word_tokenize(str3)
ps=PorterStemmer()
stemmed_words=[]

for w in wt:
    stemmed_words.append(ps.stem(w))
stemmed_words

#Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
cv1=CountVectorizer()

x_traincv=cv1.fit_transform(["Hi how are you,How are you doing","I am doing very very good","Wow that's awesome really awesome"])

x_traincv_df=pd.DataFrame(x_traincv.toarray(),columns=list(cv1.get_feature_names()))
x_traincv_df

#TF-IDF Vectorizer
tf1=TfidfVectorizer()
x_traintv=tf1.fit_transform(["Hi how are you,How are you doing","I am doing very very good","Wow that's awesome really awesome"])
x_traintv_df=pd.DataFrame(x_traintv.toarray(),columns=list(tf1.get_feature_names()))
x_traintv_df
