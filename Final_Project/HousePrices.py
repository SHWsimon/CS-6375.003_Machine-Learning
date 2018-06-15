# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 23:07:33 2018

@author: Shakti, Simon, Taishan & Wei
"""

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy import stats
from xgboost import XGBRegressor
import os
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import sys
import os.path
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import argparse
from sklearn.neural_network import MLPRegressor


training_frame = pd.read_csv('train.csv', index_col=0)
testing_frame = pd.read_csv('test.csv', index_col=0)


training_frame["SalePrice"] = pd.DataFrame({"price":training_frame["SalePrice"], "log(price + 1)":np.log1p(training_frame["SalePrice"])})
pre_processed1 = training_frame["SalePrice"]
label_train = np.log1p(training_frame.pop('SalePrice'))
pre_processed2 = pd.concat((training_frame, testing_frame), axis=0)
pre_processed2.shape

# In[2]:

#preprcessing
pre_processed2['MSSubClass'] = pre_processed2['MSSubClass'].astype(str)

#One-hot-encoder
pre_processed3 = pd.get_dummies(pre_processed2)


#resolve missing
print("missing data count:\n", pre_processed3.isnull().sum().sort_values(ascending=False).head(10))

columnMeans = pre_processed3.mean()
pre_processed3 = pre_processed3.fillna(columnMeans)
print("\nmissing data:", pre_processed3.isnull().sum().sum())

#num std
cols = pre_processed2.columns[pre_processed2.dtypes != 'object']
cols_mean = pre_processed3.loc[:, cols].mean()
cols_standarized = pre_processed3.loc[:, cols].std()
pre_processed3.loc[:, cols] = (pre_processed3.loc[:, cols] - cols_mean) / cols_standarized
final_train_frame = pre_processed3.loc[training_frame.index]
final_test_frame = pre_processed3.loc[testing_frame.index]

training_final = final_train_frame.values


# In[1]:
#PCA

X_train, X_test, label_train, y_test = train_test_split(final_train_frame, pre_processed1, test_size=0.25, random_state=42)
comp_count = 51
pca = PCA(n_components=comp_count, whiten=False).fit(final_train_frame)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
pca_train_df = pca.transform(final_train_frame)


def optionRidge():
        list_scores_test = []
        ridge = Ridge(15)
        params = [10, 15, 20, 25, 30, 40]
        for param in params:
            clf = BaggingRegressor(n_estimators=param, base_estimator=ridge)
            test_score = np.sqrt(-cross_val_score(clf, training_final, pre_processed1, cv=10, scoring='neg_mean_squared_error'))
            list_scores_test.append(np.mean(test_score))
        print("Scores:\n")
        print(min(list_scores_test))
        plt.plot(params, list_scores_test)
        plt.title("Plot of n_estimators vs CV Error")

def optionXGBoost():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    params = [
            {
                'regression__min_child_weight': [2,3,4,5,6],
                'regression__max_depth': [1,2,3,4],
                'regression__subsample': [0.8,1]
            }
        ]
    pipelining = Pipeline([
            ('regression', XGBRegressor())
        ])

    input_grid = GridSearchCV(
        pipelining,
        param_grid = params,
        cv=10, scoring='neg_mean_squared_error')

    input_grid.fit(final_train_frame, pre_processed1)
    print("best parameters:")
    print(input_grid.best_params_)
    print("root mean squared error:")
    print(np.sqrt(-input_grid.best_score_))
    outptu_scoring = ['explained_variance', 'neg_median_absolute_error', 'r2']
    result_output = cross_validate(input_grid.best_estimator_.named_steps['regression'], final_train_frame, pre_processed1, scoring=outptu_scoring, cv=10)
    print('Exaplained Variance:')
    print(np.mean(result_output['test_explained_variance']))
    print('r2:')
    print(np.mean(result_output['test_r2']))
    print('median absolute error:')
    print(np.mean(result_output['test_neg_median_absolute_error']))


def optionRandomForest():
    
        input_parameters = {
            'max_depth': range(10,16,1),
            'min_samples_leaf':range(2,40,4),
            'max_features':['sqrt', 'auto', 'log2']
        }
        ranf = RandomForestRegressor()
        clf = GridSearchCV(ranf, input_parameters, cv=10, scoring='neg_mean_squared_error')
        clf.fit(final_train_frame, pre_processed1)
        
        print("best parameters:")
        print(clf.best_params_)
        print("root mean squared error:")
        print(np.sqrt(-clf.best_score_))
        outptu_scoring = ['explained_variance', 'neg_median_absolute_error', 'r2']
        result_output = cross_validate(clf.best_estimator_, final_train_frame, pre_processed1, scoring=outptu_scoring, cv=10)
        print('Exaplained Variance:')
        print(np.mean(result_output['test_explained_variance']))
        print('r2:')
        print(np.mean(result_output['test_r2']))
        print('median absolute error:')
        print(np.mean(result_output['test_neg_median_absolute_error']))

def optionNeuralNetwork():
        parameters = {
            'hidden_layer_sizes':[30,50,100],
            'activation':['identity','logistic','tanh','relu'],
            'solver':['lbfgs'],
            'max_iter':[200],
        }
        et = MLPRegressor()
        grid = GridSearchCV(et, parameters, scoring='neg_mean_squared_error', cv=10)
        grid.fit(final_train_frame, pre_processed1)
        print("best params:")
        print(grid.best_params_)
        print("root mean squared error:")
        print(np.sqrt(-grid.best_score_))
        # print("PCA Number component:")
        # print(grid.best_estimator_.named_steps['reduce_dim'].n_components_)
        scoring = ['explained_variance', 'neg_median_absolute_error', 'r2']
        cv_result = cross_validate(grid.best_estimator_, final_train_frame, pre_processed1, scoring=scoring, cv=10)
        #print(cv_result)
        print('explained_variance:')
        print(np.mean(cv_result['test_explained_variance']))
        print('median absolute error:')
        print(-np.mean(cv_result['test_neg_median_absolute_error']))
        print('r2:')
        print(np.mean(cv_result['test_r2']))


if __name__ == '__main__':
    flag = True
    while flag:
        model = int(input('1. Ridge 2.Xgboost 3.RandomForestRegressor 4.NeuralNetwork\n'))
        if model == 1:
            optionRidge()

        if model == 2:
            optionXGBoost()

        if model == 3:
            optionRandomForest()

        if model == 4:
            optionNeuralNetwork()

        if model !=1 and model !=2 and model !=3 and model !=4:
            print("Wrong Option!!!")
            flag = False
