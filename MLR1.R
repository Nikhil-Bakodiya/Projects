getwd() #to know the directory , so as to save the file in that dierctory to read the same

data1=read.csv("C:/Users/exam.SBS/PycharmProjects/ML2/loan prediction.csv"
               ,na.strings=c("NA","NaN",""),stringsAsFactors=T)
str(data1)

data1$Credit_History=as.factor(data1$Credit_History)
str(data1)

sum(is.na(data1)) # to find out total null values in the data1

summary(data1) #To know the Na or null values available in each columns

ncol(data1) #To know the total number of columns in the data1

#
col=ncol(data1)-1
data2=data1[,2:col]

sum(is.na(data2))
summary(data2)

#Imputation (Replace the Na values with some parameters, such as mean,median, mode etc.)
install.packages("caret")
install.packages("mice") #mice=Multivariate Imputation via Chained Equations
install.packages('randomForest')

library(mice)
library(caret)
library(randomForest)
preproc=mice(data2,method="rf",m=1,maxit=1,seed=123) #m= number of data created , maxit=iterations
data3=complete(preproc)
sum(is.na(data3))

#Dummy Variables
dm=dummyVars(~.,data3,fullRank = TRUE)
X=data.frame(predict(dm,data3))
y=as.factor(data1$Loan_Status)

#Splitting the data
set.seed(10)
index=createDataPartition(y,p=0.75, list = FALSE)
X_train=X[index,]
y_train=y[index]
X_test=X[-index,]
y_test=y[-index]

#Scaling
preProcValues=preProcess(X_train, method = c("center","scale")) #fit 
X_train=predict(preProcValues,X_train) #transform
X_test=predict(preProcValues,X_test) #transform

#Model building
install.packages("e1071")
library(e1071)
model1=train(X_train,y_train,method = 'knn')

#Checking result
preds=predict(model1,X_test)
confusionMatrix(preds,y_test)

##Cross Validation
#Train Control
set.seed(10)
tc=trainControl(method = "cv",number=10)
model_knn=train(X_train,y_train,method = "knn",trControl=tc)
preds=predict(model_knn, X_test)
confusionMatrix(preds,y_test)

#Tune Length (to check the best parameters for knn)
set.seed(100)
tc=trainControl(method = 'cv',number = 10)
model_knn=train(X_train,y_train,method = 'knn',trControl = tc,tuneLength = 5)
preds=predict(model_knn,X_test)
confusionMatrix(preds,y_test)

model_knn$bestTune #5 indicates the number of iteration

#Grid Search
grid1=expand.grid(k=c(3,9,12))
model_grid=train(X_train,y_train,method = 'knn',trControl = tc,tuneGrid = grid1)
preds=predict(model_grid,X_test)
confusionMatrix(preds,y_test)

model_grid$bestTune #3 indicates the number of iteration
