import matplotlib.pyplot as pPlot
from wordcloud import WordCloud,STOPWORDS

file1=open('C:\Users\exam.SBS\Desktop\myfile.txt')
text1=file1.read()

cloud=WordCloud(background_color='white',max_words=200,stopwords=set(STOPWORDS))
cloud.generate(text1)
cloud.to_file("wordCloud.png")

