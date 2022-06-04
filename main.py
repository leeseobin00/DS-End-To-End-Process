import numpy as np
import pandas as pd
import pickle
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 16) # to show all column


# Data Inspection
df = pd.read_csv("vgsales.csv")

print("-----------------Inspection------------------")
#1. platform - histogram
print("\n--Platform--")
print(df["Platform"].value_counts())

plt.figure(figsize=[13,5])
df["Platform"] = df["Platform"].astype(str)
plt.hist(df["Platform"],bins = 32, rwidth=0.9)
plt.title ("Platform")
plt.xticks(rotation = 40)
plt.xlabel("Platform")
plt.ylabel("The number of game")

#2. years - histogram
plt.figure()
plt.hist(df["Year_of_Release"],bins=8, rwidth=0.8)
plt.title ("Years of Release")
plt.xlabel("Years")
plt.ylabel("The number of game")

#3. Genre - histogram
print("--------------Genre----------------")
print(df["Genre"].value_counts())

plt.figure(figsize=[13,5])
df["Genre"] = df["Genre"].astype(str)
plt.hist(df["Genre"],bins = 13, rwidth=0.9)
plt.title ("Genre")
plt.xlabel("Genre")
plt.ylabel("The number of game")

#4. publisher
print("\n--publisher--")
print(df["Publisher"].value_counts())

#5. sales - piechart
plt.figure()
sizes = [df['NA_Sales'].sum(), df['JP_Sales'].sum(), df['EU_Sales'].sum(), df['Other_Sales'].sum()]
labels = "NA_Sales", "JP_Sales", "EU_Sales", "Other_Sales"
plt.pie(sizes, labels=labels, autopct="%1.2f%%")

print(df.iloc[:, 5:10].describe())

#6. critical, user - histogram
fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(14,5)) #show 4 graphs in one figure
ax1.set_title('Critic_Score')
sns.distplot(df['Critic_Score'], kde=False,ax=ax1)

ax2.set_title('Critic_Count')
sns.distplot(df['Critic_Count'], kde=False,ax=ax2)

ax3.set_title('User_Score')
df['User_Score'] = np.where(df['User_Score']=='tbd',None,df['User_Score'])
sns.distplot(df['User_Score'], kde=False,ax=ax3)

ax4.set_title('User_Count')
sns.distplot(df['User_Count'], kde=False,ax=ax4)
#show a lot of null

fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(14,5)) #show 4 graphs in one figure
ax1.set_title('Critic_Score')
a = df['Critic_Score'].value_counts().sum()
b = df["Critic_Score"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax1)

ax2.set_title('Critic_Count')
a = df['Critic_Count'].value_counts().sum()
b = df["Critic_Count"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax2)

ax3.set_title('User_Score')
a = df['User_Score'].value_counts().sum()
b = df["User_Score"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax3)

ax4.set_title('User_Count')
a = df['User_Count'].value_counts().sum()
b = df["User_Count"].isnull().sum()
index = ["not null", "null"]
sns.barplot(index, [a,b],tick_label=index,ax=ax4)

#7. developer - histogram
print("\n--Developer--")
print(df["Developer"].value_counts())

#8. rating - pie chart
print("\n-- rating--")
print(df["Rating"].value_counts())
print((df['Rating']=='E').count())
plt.figure()
sizes = [3991,2961,1563,1420,15]
labels = "E", "T", "M", "E10+","etc."
plt.pie(sizes, labels=labels, autopct="%1.2f%%")

plt.show()




# Data Preprocessing
# data_type
print("-----------------Preprocessing------------------")
df = pd.read_csv("vgsales.csv")

# Data exploration
print("-------------data type -------------")
print(df.dtypes)
print()
print("-------------shape--------------")
print(df.shape)
print()
print("-------------all column ------------")
print(df.describe())
print()
print("-------------categorical data---------------")
categorical = df.dtypes[df.dtypes == "object"].index
print(categorical)
print()


# Data cleaning
# Show all nan values
# As you can see Score, Count, Developer and Rating has the most NAN values
print("-----Number of NAN -------")
print(df.isnull().sum())
print()


# First, discard all 'features' that are not related to Rating and sales volume,
# and then discard rows with NA values.
df2 = df.drop(['Name', 'Platform', 'Year_of_Release', 'Developer',
               'JP_Sales', 'Other_Sales',
               'Critic_Count', 'User_Count'], axis=1)
df2 = df2.dropna(axis=0)
idx = df2[df2['User_Score']=='tbd'].index
df2 = df2.drop(idx)
df2['User_Score'] = df2['User_Score'].astype(float, errors = 'raise')

# Global sales, critic score scatter
plt.scatter(df2['Critic_Score'], df2['Global_Sales'])
plt.xlabel("Critic_Score")
plt.ylabel("Global_Sales")
plt.show()

# Global sales, user score scatter
plt.scatter(df2['User_Score'], df2['Global_Sales'])
plt.xlabel("User_Score")
plt.ylabel("Global_Sales")
plt.show()

#show the correlation between features.
print('show the correlation between features: ')
print(df2.corr())

# User_Score, Critic_Score is droped because of the low correlation with the Global sales
df2 = df2.drop(['Critic_Score','User_Score'], axis=1)

# We are going to see sales based on rating,
#    -> excluded the rating category with too little data.
df3 = df2[(df2.Rating =='E')|(df2.Rating =='E10+')|(df2.Rating =='T')|(df2.Rating =='M')]

print("==========================")
print("new_Rating")
print(df3["Rating"].value_counts())
print()

#droped Global sales over 20 because they are outliers
df3 = df3[df3.Global_Sales < 20]

#reset the index of the df3
df3.reset_index(drop=True)

# Label Encoder
enc = LabelEncoder()

# encoding genre
encoding = pd.DataFrame(df3['Genre'])
enc.fit(encoding)
df3['Genre'] = pd.DataFrame(enc.transform(encoding))

# encoding publisher
encoding = pd.DataFrame(df3['Publisher'])
enc.fit(encoding)
df3['Publisher'] = pd.DataFrame(enc.transform(encoding))

# encoding Rating
encoding = pd.DataFrame(df3['Rating'])
enc.fit(encoding)
df3['Rating'] = pd.DataFrame(enc.transform(encoding))

# check the result of the encoding data
print("-------- result of the concatenate encoding data --------")
print(df3)

# Standard Scaler
#change the each columns' form for scaling
global_sales = np.array(df3.loc[:, ['Global_Sales']]).reshape(-1)
NA_Sales = np.array(df3.loc[:, ['NA_Sales']]).reshape(-1)
EU_Sales = np.array(df3.loc[:, ['EU_Sales']]).reshape(-1)

Scaler = StandardScaler()
#scaling Global_Sales
scaled_global_sales = Scaler.fit_transform(global_sales[:, np.newaxis]).reshape(-1)
#scaling NA_Sales
scaled_NA_Sales = Scaler.fit_transform(NA_Sales[:, np.newaxis]).reshape(-1)
#scaling EU_Sales
scaled_EU_Sales = Scaler.fit_transform(EU_Sales[:, np.newaxis]).reshape(-1)

# input the new data which is scaled into df4
df4 = pd.DataFrame({
    'Genre':df3['Genre'],
    'Publisher':df3['Publisher'],
    'Global_Sales':scaled_global_sales,
    'NA_Sales':scaled_NA_Sales,
    'EU_Sales':scaled_EU_Sales,
    'Rating': df3['Rating']
})

#show the correlation between features
cor = df4.corr()
print("------------- corr matrix ----------------")
print(cor)
print()

# Rating, Genre, Publisher are droped because of the low correlation
df4 = df4.drop(['Rating', 'Genre', 'Publisher'], axis=1)


# scaler result
print("-------- result of the concatenate scaling data --------")
print(df4)

# preprocessing result
print("--------Final DataSet shape, dtypes --------")
print(df4.shape)
print(df4.dtypes)
print()
print("------------categorical data  -------------")
categorical = df4.dtypes[df4.dtypes == "object"].index
print(categorical)
print()

print("---final dataset --------")
print(df4)
print()

# release final dataset
df4.to_csv('cleaning_vgsales.csv', index=False, header=True)




print("-----------------Analysis------------------")

#DATA ANALYSIS

from numpy import newaxis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#regeression
'''
input : data for Traing(y is target column, X is other columns)
output : trained model
definition : enter the data and make the result of trained model which is the from the Gradient Boosting Regressor
'''
def GBR(X,y):

    regressor = GradientBoostingRegressor(
        max_depth=5,
        n_estimators=500,
        learning_rate=1.0
    )
    regressor = regressor.fit(X, y)
    return regressor

#classification
'''
input: target(IsSuccess column) and other column for calculating by the KNN, n_neigbors which is the K of KNN
output: KNN model
definition : KNN doesn't need the traing but 'fit' is the process. so, I made it as a function.
'''
def KNN(X,y, n):
    classifier = KNeighborsClassifier(n_neighbors= n)
    classifier.fit(X,y)
    return classifier

#data that we will predict.
#newGame is here for scaling
newGame = pd.DataFrame({"Global_Sales": np.nan, "NA_Sales": np.nan, "EU_Sales":10}, index=[0])
print("data that will predict: ")
print(newGame)
print()

#scaling newGame
SC = StandardScaler()
SC.fit(EU_Sales[:, np.newaxis])
newGame_EU_Sales = np.array(newGame.loc[:, ['EU_Sales']]).reshape(-1)
scaled_newGame = SC.transform(newGame_EU_Sales[:, np.newaxis]).reshape(-1)

# input the scaled data into newGame
newGame = pd.DataFrame({
    "Global_Sales": np.nan,
    "NA_Sales": np.nan,
    "EU_Sales": scaled_newGame
})

print("after scaling of the new game")
print(newGame)
print()

#run the model to predict NA_Sales
X = df4.drop(columns=["Global_Sales","NA_Sales"])
y = df4["NA_Sales"]

newGameX = newGame.drop(columns=["Global_Sales","NA_Sales"])

#split X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=1,
                                                        shuffle=True)
reg_NA_Sales = GBR(X_train,y_train)

#score of the predicting NA_Sales before hyperParameter
print("score of the predict NA_Sales: ")
print(reg_NA_Sales.score(X_test, y_test))
print()

#prediction of the User_Score
print("prediction of the NA_Sales of the newGame: ")
print(reg_NA_Sales.predict(newGameX))
print()

#setting the newGame's NA_Sales
newGame["NA_Sales"] = reg_NA_Sales.predict(newGameX)
print("current newGame status(No Global_Sales yet) : ")
print(newGame)
print()

#Traing the model based on the EU_Sales, NA_Sales and predict the Global Sales
X = df4.drop(columns=["Global_Sales"])
y = df4["Global_Sales"]

newGameX = newGame.drop(columns=["Global_Sales"])

#split X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=1,
                                                        shuffle=True)
reg_Global_Sales = GBR(X_train,y_train)

#score of the predicting Global_Sales before hyperParameter
print("score of the predict Global_Sales: ")
print(reg_Global_Sales.score(X_test, y_test))
print()

#prediction of the Global_Sales
print("prediction of the Global_Sales of the newGame: ")
print(reg_Global_Sales.predict(newGameX))
print()

#setting the newGame's Global_Sales
newGame["Global_Sales"] = reg_Global_Sales.predict(newGameX)
print("current newGame status: ")
print(newGame)
print()

#add the IsSuccess colunm for predict the game is whether success or not
cutoff = df4["Global_Sales"].mean()*1.5

#global sales bigger than the cutoff = success, lower than the cutoff = failure
#and show the data frame which added the isSuccess column
df4['IsSuccess'] = ["success" if s >= cutoff else "failure" for s in df4['Global_Sales']]
print(df4)
print()

#execute model
knn_X = df4[["Global_Sales","NA_Sales","EU_Sales"]]
knn_y = df4["IsSuccess"]



newGameX = newGame[["Global_Sales","NA_Sales","EU_Sales"]]
#split the data for knn
knn_X_train, knn_X_test, knn_y_train, knn_y_test = train_test_split(knn_X, knn_y, test_size=.2,
                                                        random_state=1,
                                                        shuffle=True)

#setting the isSuccess into the newGame.
knn = KNN(knn_X_train,knn_y_train, 5)

#KNN score
print("knn score: ")
print(knn.score(knn_X_test,knn_y_test))
print()

#setting the newGame's isSuccess
newGame["IsSuccess"] = knn.predict(newGameX)

#result before hyperParameter
print("Result before hyperParameter")
print("newGame will be ",end="")
print(newGame["IsSuccess"].values)
print()




print("-----------------Evaluation------------------")

#EVALUATION

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

#initailize the newGame
newGame = pd.DataFrame({
    "Global_Sales": np.nan,
    "NA_Sales": np.nan,
    "EU_Sales": newGame['EU_Sales']
})

#hyperParameter work for predicting NA_Sales
X = df4.drop(columns=["Global_Sales","NA_Sales","IsSuccess"])
y = df4["NA_Sales"]
#split X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=1,
                                                        shuffle=True)
#setting the grid of the hyperParameters
parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.4],
              'n_estimators': [50, 100, 300, 500, 1000, 1500],
              'max_depth': [2, 3, 4, 5, 6]
              }
grid_GBR_NA_Sales = GridSearchCV(estimator=reg_NA_Sales, param_grid=parameters, cv = 3, n_jobs=-1)
grid_GBR_NA_Sales.fit(X_train, y_train)

#print the result
print(" Results from grid_GBR_NA_Sales ")
print("\n The best estimator: \n", grid_GBR_NA_Sales.best_estimator_)
print("\n The best score during all parameters: \n", grid_GBR_NA_Sales.best_score_)
print("\n The best parameters: \n", grid_GBR_NA_Sales.best_params_)
print()

#Setting the NA_Sales from the hyperParameter processed model
newGameX = newGame.drop(columns=["Global_Sales","NA_Sales"])
newGame['NA_Sales'] = grid_GBR_NA_Sales.predict(newGameX)

#current newGame status
print("current newGame status include predicted NA_Sales by the hyperparmeterd model")
print(newGame)
print()

#hyperParameter work for Global_sales
X = df4.drop(columns=["Global_Sales","IsSuccess"])
y = df4["Global_Sales"]

newGameX = newGame.drop(columns=["Global_Sales"])

#split X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=1,
                                                        shuffle=True)
parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
              'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100, 500],
              'max_depth': [2, 3, 4, 5, 6]
              }
grid_GBR_Global_Sales = GridSearchCV(estimator=reg_Global_Sales, param_grid=parameters, cv = 3, n_jobs=-1)
grid_GBR_Global_Sales.fit(X_train, y_train)

#print the result
print(" Results from grid_GBR_Global_Sales ")
print("\n The best estimator: \n", grid_GBR_Global_Sales.best_estimator_)
print("\n The best score during all parameters: \n", grid_GBR_Global_Sales.best_score_)
print("\n The best parameters: \n", grid_GBR_Global_Sales.best_params_)
print()

#Setting the Global Sales by the hyperParameter processed model
newGameX = newGame.drop(columns=['Global_Sales'])
newGame['Global_Sales'] = grid_GBR_Global_Sales.predict(newGameX)

#current newGame status
print("current newGame status include predicted Global_Sales by the hyperparmeterd model")
print(newGame)
print()

#evaluate the KNN by confusion matrix
cm = confusion_matrix(knn_y_test,knn.predict(knn_X_test))

#show the confusion martix
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Reds')
plt.xlabel('actual value')
plt.ylabel('predicted value')
plt.show()

#accuracy of the confusion matrix
print("precision, recall, f1_score of the confusion matrix which composed with KNN: ")
print(classification_report(knn_y_test, knn.predict(knn_X_test)))
print()

#newGame
newGameX = newGame[["Global_Sales","NA_Sales","EU_Sales"]]

#setting the newGame's isSuccess
newGame["IsSuccess"] = knn.predict(newGameX)

#result after hyperParameter
print("Result after hyperParameter")
print("newGame will be ",end="")
print(newGame["IsSuccess"].values)

print("End")


