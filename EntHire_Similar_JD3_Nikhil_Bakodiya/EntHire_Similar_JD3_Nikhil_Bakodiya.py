import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sea
import matplotlib.pyplot as plt
data=pd.read_csv("naukri_com-job_sample.csv")

data.shape
data.dtypes

data1=data.drop_duplicates()
data1.shape #the shape of the original dataset is equivalent to this so it shows that there is no Duplicity in the raw data

data1.isnull().sum()

#Spliting payrate into min_pay and max_pay

pay_split = data['payrate'].str[1:-1].str.split('-', expand=True)
pay_split.head()

#For min_pay
#remove space in left and right
pay_split[0] =  pay_split[0].str.strip()
#remove comma
pay_split[0] = pay_split[0].str.replace(',', '')
#remove all characters
pay_split[0] = pay_split[0].str.replace(r'\D.*', '')
#display
pay_split[0].head()

#For max_pay
#remove space in left and right
pay_split[1] =  pay_split[1].str.strip()
#remove comma
pay_split[1] = pay_split[1].str.replace(',', '')
#remove all character
pay_split[1] = pay_split[1].str.replace(r'\D.*', '')
#display
pay_split[1].head()

#checking the data type of splited numbers
pay_split[0].dtypes
pay_split[1].dtypes

#Converting them to numeric datatypes
pay_split[0] = pd.to_numeric(pay_split[0], errors='coerce')
pay_split[1] = pd.to_numeric(pay_split[1], errors='coerce')

#Concatenate min_pay and max_pay
pay=pd.concat([pay_split[0], pay_split[1]], axis=1, sort=False)

pay.rename(columns={0:'min_pay', 1:'max_pay'}, inplace=True)
pay.head()

data=pd.concat([data, pay], axis=1, sort=False)

#Spliting experience into min_experience and max_experience

experience_split = data['experience'].str[0:-1].str.split('-', expand=True)
experience_split.head()

#remove space in left and right
experience_split[1] =  experience_split[1].str.strip()
#remove comma
experience_split[1] = experience_split[1].str.replace('yr', '')
#remove all character
experience_split[1] = experience_split[1].str.replace(r'yr', '')
#display
experience_split[1].head()
#checking the data type of splited numbers
experience_split[0].dtypes
experience_split[1].dtypes

#Converting them to numeric datatypes
experience_split[0] = pd.to_numeric(experience_split[0], errors='coerce')
experience_split[1] = pd.to_numeric(experience_split[1], errors='coerce')

# Concatenate min_experience and max_experience
experience=pd.concat([experience_split[0], experience_split[1]], axis=1, sort=False)

experience.rename(columns={0:'min_experience', 1:'max_experience'}, inplace=True)
experience.head()

data=pd.concat([data, experience], axis=1, sort=False)
data.head()

#Converting the pay into average pay
data['avg_pay']=(data['min_pay'].values + data['max_pay'].values)/2
#converting the experience into average experience
data['avg_experience']=(data['min_experience'].values + data['max_experience'].values)/2

sea.stripplot(x='min_experience', y='min_pay', data=data, jitter=True)

sea.pointplot(x='min_experience', y='min_pay', data=data)

sea.stripplot(x='max_experience', y='max_pay', data=data, jitter=True)

sea.pointplot(x='max_experience', y='max_pay', data=data)

sea.pairplot(data,
             size=5, aspect=0.9,
             x_vars=["min_experience","max_experience"],
             y_vars=["min_pay"],
             kind="reg")

sea.pairplot(data,
             size=5, aspect=0.9,
             x_vars=["min_experience","max_experience"],
             y_vars=["max_pay"],
             kind="reg")

sea.jointplot(x='avg_experience', y='avg_pay', data=data,
              kind="kde",xlim={0,15}, ylim={0,1000000})

sea.stripplot(x='avg_experience', y='avg_pay', data=data, jitter=True)

sea.pointplot(x='avg_experience', y='avg_pay', data=data)
plt.tight_layout()


data[['min_pay','industry']].groupby(["industry"]).median().sort_values(by='min_pay',ascending=False).head(10).plot.bar(color='yellow')

data[['max_pay','industry']].groupby(["industry"]).median().sort_values(by='max_pay',ascending=False).head(10).plot.bar(color='lightblue')

data[['avg_pay','skills']].groupby(["skills"]).median().sort_values(by='avg_pay',ascending=False).head(10).plot.bar(color='purple')

data[['avg_pay','jobtitle']].groupby(["jobtitle"]).median().sort_values(by='avg_pay',ascending=False).head(10).plot.bar(color='orange')

replacements = {
    'joblocation_address': {
        r'(Bengaluru/Bangalore)': 'Bangalore',
        r'Bengaluru': 'Bangalore',
        r'Hyderabad / Secunderabad': 'Hyderabad',
        r'Mumbai , Mumbai,mumbai,navi mumbai': 'Mumbai',
        r'Noida': 'NCR',
        r'Delhi': 'NCR',
        r'Gurgaon': 'NCR',
        r'Delhi/NCR(National Capital Region)': 'NCR',
        r'Delhi , Delhi': 'NCR',
        r'Noida , Noida/Greater Noida': 'NCR',
        r'Ghaziabad': 'NCR',
        r'Delhi/NCR(National Capital Region) , Gurgaon': 'NCR',
        r'NCR , NCR': 'NCR',
        r'NCR/NCR(National Capital Region)': 'NCR',
        r'NCR , NCR/Greater NCR': 'NCR',
        r'NCR/NCR(National Capital Region) , NCR': 'NCR',
        r'NCR , NCR/NCR(National Capital Region)': 'NCR',
        r'Bangalore , Bangalore / Bangalore': 'Bangalore',
        r'Bangalore , karnataka': 'Bangalore',
        r'NCR/NCR(National Capital Region)': 'NCR',
        r'NCR/Greater NCR': 'NCR',
        r'NCR/NCR(National Capital Region) , NCR': 'NCR'

    }
}

data.replace(replacements, regex=True, inplace=True)
y = data['joblocation_address'].value_counts()

most_job_posting_city=data['joblocation_address'].value_counts().head()

# Bangalore has the highest number of jobs
most_job_posting_city.plot(kind = 'bar')
plt.tight_layout()
#

data.isnull().sum()#checking the null values in each columns

#Company
data["company"].value_counts().head(10)
co=data[data["company"].isnull()] #after reviewing this, the null spaces for the companies shows no relevancy so we can drop these 4 null companies complete row
# data1=data1[data1["company"].dropna()]
data=data.drop(3768,axis=0)
data=data.drop(4026,axis=0)
data=data.drop(4389,axis=0)
data=data.drop(4841,axis=0)

#Industry
data["industry"].value_counts()
q=data[data["industry"].isnull()] # After reviewing this there was only 1 row which was related to accounts and not software development so it was no use for our analysis, so we can drop it rather than imputing it
data=data.drop(18578,axis=0)

#droping the unwanted columns for our model
data=data.drop(data[["payrate","postdate","site_name","uniq_id","experience","jobdescription","jobid"]],axis=1) #these columns are not of much use to us in model building, pay rate has been already been divided into min_pay and max_pay same goes for experience as well

#Filtering the data (As we are only interested in skills which are related to software development only, so we will make a different data set of the same and apply our model)

df1=data[data['skills']=="IT Software - Application Programming"]
df2=data[data['skills']=="ITES"]
df3=data[data['skills']=="IT Software - Other"]
df4=data[data['skills']=="IT Software - Network Administration"]
df5=data[data['skills']=="IT Software - ERP"]
df6=data[data['skills']=="IT Software - QA & Testing"]
df7=data[data['skills']=="IT Software - eCommerce"]
df8=data[data['skills']=="IT Software - DBA"]
df9=data[data['skills']=="IT Software - Embedded"]
df10=data[data['skills']=="IT Software - Mobile"]
df11=data[data['skills']=="Analytics & Business Intelligence"]
df12=data[data['skills']=="IT Software - System Programming"]
df13=data[data['skills']=="IT Software - Telecom Software"]
df14=data[data['skills']=="IT Software - Client/Server Programming"]
df15=data[data['skills']=="IT Software - Systems"]
df16=data[data['skills']=="IT Software - Middleware"]
df17=data[data['skills']=="IT Software - Mainframe"]

df=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17],ignore_index=True)

df.isnull().sum()
#filling the null places:

#Education
df["education"].value_counts()
sea.barplot(x="skills",y="max_pay",data=data.head(10))
df['education'].fillna(data['education'].mode()[0],inplace=True)

#joblocation_address
df["joblocation_address"].value_counts()
df['joblocation_address'].fillna(df['joblocation_address'].mode()[0],inplace=True)

#numberofpositions
df["numberofpositions"].value_counts()
df['numberofpositions'].fillna(df['numberofpositions'].median(),inplace=True)

#min_pay
df["min_pay"].fillna(df["min_pay"].median(),inplace=True)

#max_pay
df["max_pay"].fillna(df["max_pay"].median(),inplace=True)

#min_experience
df["min_experience"].fillna(df["min_experience"].median(),inplace=True)

#max_experience
df["max_experience"].fillna(df["max_experience"].median(),inplace=True)

#avg_pay
df["avg_pay"].fillna(df["avg_pay"].median(),inplace=True)

#avg_experience
df["avg_experience"].fillna(df["avg_experience"].median(),inplace=True)

#

df["skills"].value_counts()
df.shape

#Converting the textual data into machine readable numbers
df["company"].value_counts()
df["education"].value_counts()
df["industry"].value_counts()
df["joblocation_address"].value_counts()
df["jobtitle"].value_counts()
df["skills"].value_counts()

#Converting the above columns into str type so that label encoding can be performed on them
df['company'] = df['company'].astype(str)
df['education'] = df['education'].astype(str)
df['industry'] = df['industry'].astype(str)
df['joblocation_address'] = df['joblocation_address'].astype(str)
df['jobtitle'] = df['industry'].astype(str)
df['skills'] = df['industry'].astype(str)

df.info
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['company']=le.fit_transform(df['company'])
df['education']=le.fit_transform(df['education'])
df['industry']=le.fit_transform(df['industry'])
df['joblocation_address']=le.fit_transform(df['joblocation_address'])
df['jobtitle']=le.fit_transform(df['jobtitle'])
df['skills']=le.fit_transform(df['skills'])


#Putting values of df into variable X
X=df.values
#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#Applying Kmeans Clustering

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
list1=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X)
    list1.append(kmeans.inertia_) #inertia= WCSS
plt.plot(range(1,11),list1,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  #WCSS= Within cluster sum of squares

#Fitting K-means to the dataset
kmeans=KMeans(n_clusters=7,random_state=10)
y_kmeans=kmeans.fit_predict(X)

df['kmeans']=y_kmeans
df.replace({'kmeans':{0:'Red',1:'Blue',2:'Green',3:'cyan',4:'magenta',5:'purple',6:'pink'}})
df["kmeans"].value_counts()

#Visualizing the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='magenta',label='Cluster 5')
plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],s=100,c='purple',label='Cluster 6')
plt.scatter(X[y_kmeans==6,0],X[y_kmeans==6,1],s=100,c='pink',label='Cluster 7')


plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of job details')

kmeans.cluster_centers_

#Heat Map
sea.heatmap(df.corr(),cmap='coolwarm',annot=True,linewidths=0.50)
plt.title("Heat Map")
#Lets Consider skills as our target variable as skills is having high correlation with other factors and apply different supervised learning, to predict the required skills for a particualr position

df1=df.drop(df[["skills","kmeans"]],axis=1)

X = df1.values
Y = df["skills"].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.25,random_state=10)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Linear Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_pred=reg.predict(X_test)

#To check how good the model is:

#Coefficient of determination R
print('Accuracy on training data:{:.3f}'.format(reg.score(X_train,Y_train))) #100%
print('Accuracy on test data: {:.3f}'.format(reg.score(X_test,Y_test)))  #100%
#print('R2 Score {:.3f}'.format(r2_score(y_test,y_pred)))

#To find the coefficient and the intercept:
print(reg.coef_)
print(reg.intercept_)

#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

print("Training Accuracy :{:.3f}".format(classifier.score(X_train,Y_train))) #84.1%
print("Testing Accuracy :{:.3f}".format(classifier.score(X_test,Y_test))) # 82%

#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7)
classifier.fit(X_train, Y_train)

print("Training Accuracy :{:.3f}".format(classifier.score(X_train,Y_train))) #88.4%
print("Testing Accuracy :{:.3f}".format(classifier.score(X_test,Y_test))) #85.1%

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

print("Training Accuracy :{:.3f}".format(classifier.score(X_train,Y_train))) #96.9%
print("Testing Accuracy :{:.3f}".format(classifier.score(X_test,Y_test))) #95.4%

#Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

print("Training Accuracy :{:.3f}".format(classifier.score(X_train,Y_train))) #91.3%
print("Testing Accuracy :{:.3f}".format(classifier.score(X_test,Y_test))) #89.2%

#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

print("Training Accuracy :{:.3f}".format(classifier.score(X_train,Y_train))) #100%
print("Testing Accuracy :{:.3f}".format(classifier.score(X_test,Y_test))) #99.8%

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

print("Training Accuracy :{:.3f}".format(classifier.score(X_train,Y_train))) #100%
print("Testing Accuracy :{:.3f}".format(classifier.score(X_test,Y_test))) #99.99%

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

print("Training Accuracy :{:.3f}".format(classifier.score(X_train,Y_train))) #100%
print("Testing Accuracy :{:.3f}".format(classifier.score(X_test,Y_test))) #97.5%

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
cm
#Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(classifier,X,y,cv=5)
#accuracies=cross_val_score(log_reg,X,y,cv=10)
print('{:.3f}'.format(accuracies.mean())) #97.8 %

#Gridsearch (it is used to find the best estimator,leafs,max depth etc.)
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators': [10,20,30],'max_depth':[3,5,7],'min_samples_leaf': [2,5,10]}

cv_rfc=GridSearchCV(estimator=classifier,param_grid=param_grid,cv=5)
cv_rfc.fit(X_train,Y_train)
print(cv_rfc.best_score_.round(5))
y_pred=cv_rfc.predict(X_test)

print('Testing Accuracy:{:.3f}'.format(cv_rfc.score(X_test,Y_test))) # 95.2%

print(cv_rfc.best_params_) #{'max_depth': 7, 'min_samples_leaf': 2, 'n_estimators': 30}

# Linear Regression, Naive Bayes and Decision Tree were leading to over fiting
# The testing result of all are mentioned below:
# 1. Logistic Regression — 82%
# 2. Nearest Neighbor — 85.1%
# 3. Support Vector Machines — 95.4%
# 4. Kernel SVM — 89.2%
# 5. Naive Bayes — 99.8% (Overfitting)
# 6. Decision Tree Algorithm — 99.9% (Overfitting)
# 7. Random Forest Classification — 95.2%


#Visualization
import numpy as np
#Feature importance (which feature, i.e. column is the most important in this model)
x=df1
n_features=x.shape[1]
plt.barh(range(n_features),classifier.feature_importances_,align='center')
plt.yticks(np.arange(n_features),x.columns)
plt.xlabel('Feature Importance')
plt.tight_layout()

imp=list(zip(np.round(classifier.feature_importances_,2),x.columns))
imp.sort(reverse=True)
print(imp) #Weighatges of feature which are contributing in the model (Weightage of job title is the highest

#Pairplot among the most important features which are contributing 10 or more than 10% to our dependent variable
sea.pairplot(df[["jobtitle","industry","education",]])
plt.tight_layout()
