# House Prices: Advanced Regression Techniques
Implement the  preprocessing techniques to fix the raw dataset, and use the fixed dataset to build four different models Xgboost, Random Forest, Linear Regression, Neural Network. Finally, input the test dataset to predict sales prices.

## Files included:
1. HousePrice.py - the python script for House Prices predict using regression
2. test.csv - testing dataset
3. train.csv - training dataset

## Step to run the code:
Excute the follwoing command to run the python program on the command line
1. python3 HousePrices.py
    e.g.:

        python3 HousePrices.py
        
2.  Enter number to choose which mode to run
    e.g.:
    
        1.Ridge 2.Xgboost 3.RandomForestRegressor 4.NeuralNetwork

## Output
1. best params:{''}
2. root mean squared error:
3. explained_variance:
4. median absolute error:
5. r2:

## Environment
for xgboost, if you are going to run it on Windows, make sure you have installed py-xgboost and then comment `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` please.


## Tips
If you just want to test the code and run it fast, simply use the best parmameter. e.g:
for xgboost, replace
```
params = [
{
'regression__min_child_weight': [2,3,4,5,6],
'regression__max_depth': [1,2,3,4],
'regression__subsample': [0.8,1]
}
]
```
with

```
params = [
{
'regression__min_child_weight': [5],
'regression__max_depth': [3],
'regression__subsample': [0.8]
}
]
```
Otherwise, it may be really time consuming. You can find best parameters of different models from our report.

