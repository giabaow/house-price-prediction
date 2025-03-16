# House Price Predictions
From the Intermediate Machine Learning course, after learning how to clean and handle missing data from a raw dataset before applying a machine learning model, I created a file to practice what I have learned. Here are some important notes I wanted to record.\
\
At first, to make the prediction more easier, i deleted rows contain null value in "Sale Price" column.\
```
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
```
Then, I selected only the "category" column, excluding those with an object data type.\
```
X = X.select_dtypes(exclude=['object'])
X_test = X_test.select_dtypes(exclude=['object'])
```
After that, I divided "train" dataset into validate dataset and training dataset\
```
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
```

## Handle missing value
After checking the quality of the data set. I realized there are some missing value in the trainning data set. \
\
Therefore i used a missing value handling method which is SimpleImputer. SimpleImputer helps me to generate a new value for missing position which is the mean or median of that column. 

## Forest Regression Model
I built a model function base on Forest Regression Model to predict value and a function to measure the average magnitude of errors between predicted and actual values which is mean_absolute_error. 
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def mean_absolute_error_model(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae = (mean_absolute_error(y_valid, preds))
    return mae
```

## Visualization
After using mean value to handle missing value, my ```mean_absolute_error``` function return ```18250.6``` which is higher than the value if I just drop all the row contain missing value.

Therefore, I created some visualizations to determine whether the mean or median is better
\
![image alt](https://github.com/giabaow/house-price-prediction/blob/a77d8888713765a8fcf2427317f8b5fb0dca6fd3/img1.png)
![image alt](https://github.com/giabaow/house-price-prediction/blob/8c56caf4f2aa935285a5057a18a6e0ea8c6c7ebb/img2.png)

From the visualization we can see most of column with missing data is skewness. Therefore, using "median" imputation is a better choice more then "mean" method

## Final Model
I created a final model with median Simple Imputer method.\
Finally, I have final prediction results \
![image alt](https://github.com/giabaow/house-price-prediction/blob/8cafa6217fffab5a7352c825d3f80c5e9e0c7185/img3.png)








