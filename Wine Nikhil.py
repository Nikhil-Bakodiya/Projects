import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import sklearn

data=pd.read_csv("C:\\Users\exam.SBS\Desktop\wineQualityReds.csv")
data.isnull().sum()
sea.pairplot(data[['residual.sugar','free.sulfur.dioxide','pH']])
sea.heatmap(data.corr(),cmap='coolwarm',annot=True,linewidths=0.50)
plt.tight_layout()

data[['residual.sugar','pH']].corr()
data[['citric.acid','pH']].corr()
data[['chlorides','pH']].corr()
data[['free.sulfur.dioxide','pH']].corr()
data[['total.sulfur.dioxide','pH']].corr()
data[['sulphates','pH']].corr()

df=data.drop(data[['Unnamed: 0','quality']],axis=1)



df= pd.DataFrame(data,columns=['fixed.acidity','total.sulfur.dioxide','citric.acid','density','free.sulfur.dioxide','volatile.acidity','chlorides','sulphates','alcohol'])
X=df.values

y=data['pH'].values




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=10)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=np.round(pca.explained_variance_ratio_,4)
explained_variance

from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score

reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)


#To check how good the model is:

#Coefficient of determination R
print('Accuracy on training data:{:.3f}'.format(reg.score(X_train,y_train)))
print('Accuracy on test data: {:.3f}'.format(reg.score(X_test,y_test)))
#print('R2 Score {:.3f}'.format(r2_score(y_test,y_pred)))

#To find the coefficient and the intercept:
print(reg.coef_)
print(reg.intercept_)