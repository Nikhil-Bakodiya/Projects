# Kmeans clustering


data1=read.csv("C:/Machine Learning/loan prediction.csv",na.strings = c("NA","NAN"),
              stringsAsFactors = T)
str(data1)
data1$Credit_History=as.factor(data1$Credit_History)
str(data1)


sum(is.na(data1))
summary(data1)

data1$TotalIncome=data$ApplicantIncome+data$CoapplicantIncome

data2=data1[c("TotalIncome","LoanAmount")]

install.packages("caret")
library(caret)
install.packages("mice") #mice=Multivariate Imputation via Chained Equations
library(mice)
install.packages('randomForest')
library(randomForest)

preproc=mice(data2,method="rf",m=1,maxit=1,seed=123) #m= number of data created , maxit=iterations
data3=complete(preproc)
sum(is.na(data3))

#--------------------------------------------------

prePocValues=preProcess(data3, method=c("center","scale"))
dataKM=predict(prePocValues,data3)
set.seed(20)
clust=kmeans(dataKM,5)

clust$tot.withinss
k=seq(1,10)
wss=vector()
for(i in k){
  clust1=kmeans(dataKM,i)
  print(clust$tot.withinss)
  wss[i]=clust1$tot.withinss
}

plot(k,wss,type="b")

dataKM$cluster= as.factor(clust$cluster)
#dataKM=dataKM[C("TotalIncome","LoanAmount")]
library(ggplot2)
ggplot(dataKM,aes(x=TotalIncome,y=LoanAmount))+geom_point(size=2,
                  aes(color=factor(cluster)))



# -------------------------------------------------------------------

#h-clust
edist=dist(dataKM,method="euclidean")
hclust1=hclust(edist,method="ward.D")
hclust1
plot(hclust1)

#Cuttree at desire no of cluster
hclust2=cutree(hclust1,5)
hclust2







