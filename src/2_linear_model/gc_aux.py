#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from sklearn.metrics import mean_absolute_error, mean_squared_error

def r2_score(y_true, y_pred):
    corrcoef_matrix = np.corrcoef(y_true, y_pred)
    corrcoef = corrcoef_matrix[0, 1]
    return corrcoef**2
    

def ProcessData(df: pd.DataFrame) -> pd.DataFrame:
    
    df_res = df.copy()
    
    df_res = df_res.astype({col: 'str' for col in df_res.columns if df_res[col].dtype != 'object'})
    
    a_loc = df_res.columns.get_loc('SMILES')
    b_loc = df_res.columns.get_loc('Const_Value')
    c_loc = df_res.columns.get_loc('CH3')
    end_loc = len(df_res.columns)
    
    df_res = df_res.iloc[:, list(range(a_loc, a_loc+1)) + list(range(b_loc, b_loc+1)) + list(range(c_loc, end_loc))]
    df_res = df_res[df_res['CH3'] != 'No']

    #remove all the rows(compounds) where their const values are null
    df_res = df_res.loc[:, (df_res != 0).any(axis=0)]
    
    df_res['Const_Value'] = df_res['Const_Value'].str.replace(',', '.', regex=True).astype(float)

    df_values = df_res.loc[:,'Const_Value':]
    for column in df_values.columns:
        df_res[column] = pd.to_numeric(df_res[column], errors='coerce')
        
    return df_res
    
def RemoveNullGroups(df: pd.DataFrame) -> pd.DataFrame:
    
    df.loc['sum'] = df.sum()
    df = df.loc[:, df.loc['sum'] != 0]
    
    df = df.drop('sum')
    
    return df
    
def plot_learning_curve(estimator, title, X, y, ylim, cv=5, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="neg_mean_absolute_error")
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt
    
def plot_learning_curve_with_vt_set(estimator, title, X_train, y_train, X_test, y_test, ylim):
    train_sizes=np.linspace(0.1, 1.0, 5) * len(X_train)
    train_sizes = train_sizes.astype(int)
    plt.figure()
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_scores = []
    test_scores = []
#     train_scores_mean = []
#     train_scores_std = []
#     test_scores_mean = []
#     test_scores_std = []

    for train_size in train_sizes:
        X_train_subset = X_train[:train_size]
        y_train_subset = y_train[:train_size]

        estimator.fit(X_train_subset, y_train_subset)
        y_train_pred = estimator.predict(X_train_subset)
        y_test_pred = estimator.predict(X_test)

        train_score = mean_absolute_error(y_train_subset, y_train_pred)
        test_score = mean_absolute_error(y_test, y_test_pred)

        train_scores.append(train_score)
        test_scores.append(test_score)
        
#         train_scores_mean.append(np.mean(train_score))
#         train_scores_std.append(np.std(train_score))
#         test_scores_mean.append(np.mean(test_score))
#         test_scores_std.append(np.std(test_score))

    plt.grid()
#     plt.fill_between(train_sizes, -np.array(train_scores_mean) - np.array(train_scores_std),
#                      -np.array(train_scores_mean) + np.array(train_scores_std), alpha=0.1, color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores, 'o-', color="r", label='Training score')
    plt.plot(train_sizes, test_scores, 'o-', color="g", label='Test score')
    
    plt.legend(loc="best")
    return plt

