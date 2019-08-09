getwd()
install.packages("tm")
install.packages('wordcloud')
install.packages('RColorBrewer')
library(tm)
library(wordcloud)
library(RColorBrewer)

setwd("E:\\Nikhil\\R\\TM")
doc1=readLines("mydoc.txt")
doc1

corpus=Corpus(VectorSource(doc1))
inspect(corpus)

#converts all text to lower case
corpus1=tm_map(corpus,tolower)
inspect(corpus1)
#remove white space
corpus1=tm_map(corpus1,removeNumbers)
#removes punctuation
corpus1=tm_map(corpus1,removePunctuation)
#removes common words like "a","the" etc
corpus1=tm_map(corpus1,removeWords,stopwords("en"))

all_stop=c("even","enough",stopwords("en"))
corpus1=tm_map(corpus1,removeWords,all_stop)

inspect(corpus1)
inspect(corpus1[1:2])

#From corpus
wordcloud(corpus1,min.freq = 1,random.order = F,colors = brewer.pal(8,"Set2"))

dtm=TermDocumentMatrix(corpus1) #turns the corpus into a document
inspect(dtm)
corpMat=as.matrix(dtm)

dtm1=DocumentTermMatrix(corpus1)
inspect(dtm1)
