import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

data=pd.read_csv("C:\\Users\exam.SBS\PycharmProjects\ML2\CompanyData.csv")

#Scatter plot to view SAles and Profit at different Types(H,M,L)
plt.scatter(data['Sales'],data['Profit'],data['Type']=='H',c='red')
plt.scatter(data['Sales'],data['Profit'],data['Type']=='M',c='blue')
plt.scatter(data['Sales'],data['Profit'],data['Type']=='L',c='green')
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend()

#Box plot
sea.boxplot(x='Type',y='Sales',data=data)
sea.boxplot(x='Type',y='Profit',data=data)
sea.boxplot(x='Type',y='MarketCap',data=data)
sea.boxplot(x='Type',y='EPS',data=data)

#Correlation


#Pairplot
sea.pairplot(data[['Sales','Profit','EPS','MarketCap','Type']])



#Applying KNN is to find out the H,M,L
df=data.drop(['Co_Code','Type'],axis=1)

X=df.values
y=data['Type'].values

#Spliting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=100)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#KNN
from sklearn.neighbors import KNeighborsClassifier

Knn=KNeighborsClassifier(n_neighbors=5)
Knn.fit(X_train,y_train)
y_pred=Knn.predict(X_test)


print("Training Accuracy:{:.3f}".format(Knn.score(X_train,y_train)))

print('Testing Accuracy:{:.3f}'.format(Knn.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#To find out the best k values

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

neighbors=range(1,11)
k_score = dict()
for n in neighbors:
    knn1 = KNeighborsClassifier(n_neighbors=n)
    clf1 = make_pipeline(sc, knn1)
    accuracies1 = cross_val_score(clf1, X, y, cv=10)
    k_score.update({n: accuracies1.mean()})
    print('{:.3f}'.format(accuracies1.mean()))  # print the # of accuracies

print(k_score)

import matplotlib.pyplot as plt
plt.plot(neighbors,k_score.values())
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors')