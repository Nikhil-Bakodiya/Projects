import pandas as pd
import seaborn as sea

data=pd.read_csv("D:\Analytics (sem3)\Machine Learning\Assignment\Facebook_data.csv")

#q1

sea.boxplot(x='Type',y='Like',data=data)# The likes on status are more as compared to photos
sea.boxplot(x='Share',y='Interactions',data=data)
sea.distplot(data['Share'],bins=50)
sea.distplot(data['Interactions'],bins=50)

#q2
#In case of Label Encoder if there are different numbers in the same column, the model will misunderstand the data to be in some
#kind of order, 0 < 1 < 2. But this isn’t the case at all. To overcome this problem, we use One Hot Encoder.
#What one hot encoding does is, it takes a column which has categorical data, which has been label encoded, and then splits the
# column into multiple columns. The numbers are replaced by 1s and 0s, depending on which column has what value.
#Situation where label encoding is relevant is where when the categorical values are only of two variables suuch as 'yes' or 'no', etc.

#q3
df=pd.get_dummies(data['Type'])

#q4
data.isnull().sum()

data["Interactions"].value_counts()
data["Share"].value_counts()

data["Interactions"].fillna(data["Interactions"].median(),inplace=True)
data["Share"].fillna(data["Share"].median(),inplace=True)
data.isnull().sum()

#q5 Applying regression
data["Like"].value_counts()
data["Share"].value_counts()

X=data['Share'].values
y=data['Like'].values
X=X.reshape(-1,1)
#splitting the data:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

#applying linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)


#Coefficient of determination R^2
print('Accuracy on training data:{:.3f}'.format(reg.score(X_train,y_train)))
print('Accuracy on test data: {:.3f}'.format(reg.score(X_test,y_test)))