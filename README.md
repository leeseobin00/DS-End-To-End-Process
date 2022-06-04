# DS-End-To-End-Process
## 1. Introduction
It provides a judgment indicator on whether or not it is possible to enter the global market to the decision makers of game companies that are currently earning a lot of profits in their own country.

1. Predict NA_Sales with EU_Sales
2. Predict Global_Sales with EU_Sales and NA_Sales
3. Cutoff is applied with Global_Sales to determine success or failure and return

## 2. Dataset 
We use [**Video Game Sales with Ratings**](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings) data set. 

This data set simply extends the number of variables with another web scrape from Metacritic.

Alongside the fields: Name, Platform, YearofRelease, Genre, Publisher, NASales, EUSales, JPSales, OtherSales, Global_Sales. 

## 3. Data Exploration
Dataset - Columns analysis     
16 Columns, 16719 Data    
10 Numerical Data & 6 Categorical Data
![image](https://user-images.githubusercontent.com/70849467/171990145-74c0cf3f-5656-43b1-a1bb-9889a5e434a9.png)

## 4. Data Preprocessing
1. Check Missing Data
2. Ignore the data - 'ANY' Row with NAN
3. Ignore the Feature - irrelevant feature
4. Data Filtering - Rating with small data
5. Encoding Categorical Data
6. Data Filtering
7. Normalization - Standard Scaling
8. Make Clean Dataset

## 5. Modeling
1. Create New Game data only has a EU_Sales
2. Predict the NA_Sales by the Gradient Boostring Regressor
3. Predict the Global_Sales by the EU_Sales and NA_Sales
4. Put the isSuccess column into the data frame by the cut off

## 6. Evalution
1. Initalize the newGame
2. HyperParameter work for predicting NA_Sales using Gradient Boosting Regressor
3. Split Train set and Test set
4. Set the grid of the hyperParameters
5. Set the NA_Sales from the hyperParameter processed model
6. HyperParameter work for Global_Sales using Gradient Boosting Regressor
7. Split Train set and Test set
8. Evaluate the KNN by Confustion Matrix
9. Accuracy of the Confusion matrix
10. Set the newGame's isSuccess
11. Show result after hyperParameter
