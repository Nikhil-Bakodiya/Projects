import pandas as pd
data=pd.read_csv("C:\\Users\\exam.SBS\\Desktop\\Titanic\\train.csv")
import matplotlib.pyplot as plt
import seaborn as sea
# data.drop(data[["Passenger","Ticket","Cabin"]],axis=1)
sea.distplot(data['Age'],bins=50)
data.isnull().sum()
data['Age'].value_counts()

data['Age'].fillna(data['Age'].median(),inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)


plt.scatter(x='Age',y='Fare',data=data)

# bb=pd.crosstab(data['Gender'],data['Age'])
# bb
# bb.plot.bar(stacked=True)


fig,ax=plt.subplots(2,2)
a=sea.barplot(x=data['Pclass'],y=data['Survived'],ax=ax[0,0]) #1st class
#Which city people survived the most
b=sea.barplot(x=data['Embarked'],y=data['Survived'],ax=ax[0,1]) #C:Cherbourg
#Which Gender survived the most
c=sea.barplot(x=data['Gender'],y=data['Survived'],ax=ax[1,0]) #Female

d=sea.boxplot(x='Survived',y='Age',data=data,ax=ax[1,1])

plt.show()

#Applying the Logistic Regression after preprocessing

#Preprocessing
df=data.drop(data[["PassengerId","Name","Ticket","Cabin","Survived"]],axis=1)
df=pd.get_dummies(df,columns=['Pclass','Gender','SibSp','Parch','Embarked'])

X=df.values
y=data['Survived'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=10)

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

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(log_reg,X,y,cv=10)
#accuracies=cross_val_score(log_reg,X,y,cv=5)
print('{:.3f}'.format(accuracies.mean())) #This validation is without the standardised values i.e. it is on the original values that we don't want

#Pipeline (This is used to make space for standardization and log_reg togather in one variable and use it togather on validation)
from sklearn.pipeline import make_pipeline
clf=make_pipeline(sc,log_reg) # Variable(clf) assigned to store sc(standard scale) and log_reg (logarithmic regression) function
accuracies=cross_val_score(clf,X,y,cv=10)
print('{:.3f}'.format(accuracies.mean()))

##

#PCA (Principal components analysis)
import numpy as np
from sklearn.decomposition import PCA
pca=PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=np.round(pca.explained_variance_ratio_,4)
explained_variance

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)

print("Training Accuracy:{:.3f}".format(log_reg.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(log_reg.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)