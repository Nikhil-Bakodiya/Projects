import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# mystopwords
import nltk
from nltk.tokenize import word_tokenize

df_amazon=pd.read_csv("C:\\Users\exam.SBS\Desktop\\amazon_alexa.tsv",sep="\t")
df_amazon.shape
df_amazon.feedback.value_counts()

import string
punctuations=string.punctuation

#Stop Words
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))

mystopwords=set(['...','also','would','still','dot'])

def stopfun(wt1):
    filtered1=[]
    for w in wt1:
        if w not in stop_words:
            if w not in string.punctuation:
                if w not in mystopwords:
                    if len(w)>2:
                        filtered1.append(w)
    return filtered1

#Lemma
from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()

def lemfun(wt1):
    lem_words=[]
    for w in wt1:
        lem_words.append(lem.lemmatize(w,'v'))
    return lem_words

def tokenizer(sentence):
    #Tokenization
    mytokens=word_tokenize(sentence)
    #Removing stop words & punctuation
    mytokens=stopfun(mytokens)
    #mytokens=[word for word in mytokens if word not in stop_words
    #Lemmatization
    mytokens=lemfun(mytokens)
    return mytokens

# str10="which has five points?"
# tokenizer(str10)

X=df_amazon['verified_reviews'] #the features we want to analyze
y=df_amazon['feedback'] #the labels, or answers, we want

#CountVectorizer
cv_vector=CountVectorizer(tokenizer=tokenizer,max_df=.90,min_df=.05)
x_train_cv=cv_vector.fit_transform(X)
x_train_cv_df=pd.DataFrame(x_train_cv.toarray(),columns=list(cv_vector.get_feature_names()))

#TfidfVectorizer

tf_vector=TfidfVectorizer(tokenizer=tokenizer,max_df=.90,min_df=.05)
x_train_tf=tf_vector.fit_transform(X)
x_train_tf_df=pd.DataFrame(x_train_cv.toarray(),columns=list(tf_vector.get_feature_names()))

#Split the data into a model of 25% and 75%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_train_tf_df,y,test_size=.25,random_state=10)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)

print("Training Accuracy:{:.3f}".format(log_reg.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(log_reg.score(X_test,y_test)))