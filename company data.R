data1=read.csv("C:/Users/exam.SBS/PycharmProjects/ML2/CompanyData.csv")
str(data1)
summary(data1)

ncol(data1)

#
col=ncol(data1)-1
data2=data1[,2:col]
#
X=data.frame(data2)
y=as.factor(data1$Type)

#
library(caret)
#Split
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
