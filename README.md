# House Price Predictions
From the Intermediate Machine Learning course, after learning how to clean and handle missing data from a raw dataset before applying a machine learning model, I created a file to practice what I have learned. Here are some important notes I wanted to record.
At first, to make the prediction more easier, i deleted rows contain null value in "Sale Price" column.
Then, I selected only the "category" column, excluding those with an object data type.
After that, I divided "train" dataset into validate dataset and training dataset

##Handle missing value
Àfter checking the quality of the data set. I realized there are some missing value in the trainning data set. Therefore i used a missing value handling method which is SimpleImputer. SimpleImputer helps me to generate a new value for missing position which is the mean or median of that column. 
I built a model function and a function to measure the average magnitude of errors between predicted and actual values. 


