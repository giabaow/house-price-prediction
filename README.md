# House Price Predictions
From the Intermediate Machine Learning course, after learning how to clean and handle missing data from a raw dataset before applying a machine learning model, I created a file to practice what I have learned. Here are some important notes I wanted to record.
At first, to make the prediction more easier, i deleted rows contain null value in "Sale Price" column.
```
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
```
Then, I selected only the "category" column, excluding those with an object data type.
```
X = X.select_dtypes(exclude=['object'])
X_test = X_test.select_dtypes(exclude=['object'])
```
After that, I divided "train" dataset into validate dataset and training dataset
```
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
```

## Handle missing value
After checking the quality of the data set. I realized there are some missing value in the trainning data set. Therefore i used a missing value handling method which is SimpleImputer. SimpleImputer helps me to generate a new value for missing position which is the mean or median of that column. 
I built a model function and a function to measure the average magnitude of errors between predicted and actual values. 


