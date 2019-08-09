#SVM
install.packages("kernlab")
library(kernlab)
model_svm=train(X_train,y_train,method = 'svmLinear',trControl = tc)
preds=predict(model_svm,X_test)
confusionMatrix(preds,y_test)

#Tree
model_tree=train(X_train,y_train,method = 'rpart',trControl = tc)
preds=predict(model_tree,X_test)
confusionMatrix(preds,y_test)

#Random Forest
model_rf=train(X_train,y_train,method = 'rf',trControl = tc)
preds=predict(model_rf,X_test)
confusionMatrix(preds,y_test)

#logistic reg
model_log=train(X_train,y_train,method = 'glm',trControl = tc)
preds=predict(model_log,X_test)
confusionMatrix(preds,y_test)

#xgbTree
model_xgb=train(X_train,y_train,method = 'xgbTree',trControl = tc)
preds=predict(model_xgb,X_test)
confusionMatrix(preds,y_test)