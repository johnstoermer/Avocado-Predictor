#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.graph_objs as go

def load_df(f):
    df = pd.read_csv(f, index_col=0)
    df = df[['AveragePrice', 'Date', 'Total Volume', 'type', 'year', 'region']]
    df['Date'] = df['Date'].apply(lambda x: int(x[5:7]))
    df['type'] = df['type'].apply(lambda x: 0 if x == 'conventional' else 1)
    df = df.rename(columns={'Date': 'month'})
    df = pd.get_dummies(df)
    return df

def train_it_2layer(Xtrain, ytrain, lr, reg):
    avocado_model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(58, input_dim=58, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=reg)),
      tf.keras.layers.Dense(58, input_dim=58, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=reg)),
      tf.keras.layers.Dense(1)
    ])
    avocado_model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse', metrics=['mae'])
    history = avocado_model.fit(Xtrain, ytrain, epochs=2)
    return avocado_model, history

def train_it_linear(Xtrain, ytrain, lr, reg):
    avocado_model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(58, input_dim=58, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(l=reg)),
      tf.keras.layers.Dense(1)
    ])
    avocado_model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse', metrics=['mae'])
    history = avocado_model.fit(Xtrain, ytrain, epochs=2)
    return avocado_model, history

def test_it(model, Xtest, ytest):
    prediction = model.predict(Xtest)
    train_error =  (ytest - prediction) ** 2
    return np.mean(train_error)

'''
Shuffle the dataset randomly.
Split the dataset into k groups
For each unique group:
Take the group as a hold out or test data set
Take the remaining groups as a training data set
Fit a model on the training set and evaluate it on the test set
Retain the evaluation score and discard the model
Summarize the skill of the model using the sample of model evaluation scores
'''
def kfold_it(df, lr, reg, linear=False): #k = 3
    df = df.sample(frac=1)
    folds = np.array_split(df, 3)
    fold_mses = []
    fold_hists = []
    for i in range(3):
        test_df = folds[i]
        train_df = pd.concat([folds[j] for j in range(3) if j != i], axis=0)
        if linear:
            model, hist = train_it_linear(train_df.values[:,1:], train_df.values[:,0], lr, reg)
        else:
            model, hist = train_it_2layer(train_df.values[:,1:], train_df.values[:,0], lr, reg)
        fold_mse = test_it(model, test_df.values[:,1:], test_df.values[:,0])
        print('Fold ' + str(i) + ' MSE: ' + str(fold_mse))
        fold_hists.append(hist.history['mean_absolute_error'])
        fold_mses.append(fold_mse)
    return np.mean(fold_mses), np.std(fold_mses), np.mean(fold_hists, axis=0) ** 2

if __name__ == "__main__":
    df = load_df('avocado.csv')
    #test learning rates
    lrs = [0.001, 0.003, 0.005]
    #2layer
    hists = []
    for lr in lrs:
        a, b, hist = kfold_it(df, lr, 0.1, linear=False)
        hists.append(hist)
    fig = go.Figure()
    X = list(range(100))
    for i in range(len(hists)):
        fig.add_trace(go.Scatter(x=X, y=np.log(hists[i]),
                        mode='lines',
                        name=str(lrs[i]) + ' learning rate'))
    fig.layout.update(
        title='Learning Rate effect on MSE',
        xaxis_title='Epoch',
        yaxis_title='log(MSE)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.write_image('fig1.pdf')
    #linear
    hists = []
    for lr in lrs:
        a, b, hist = kfold_it(df, lr, 0.1, linear=True)
        hists.append(hist)
    fig = go.Figure()
    X = list(range(100))
    for i in range(len(hists)):
        fig.add_trace(go.Scatter(x=X, y=np.log(hists[i]),
                        mode='lines',
                        name=str(lrs[i]) + ' learning rate'))
    fig.layout.update(
        title='Learning Rate effect on MSE',
        xaxis_title='Epoch',
        yaxis_title='log(MSE)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.write_image('fig2.pdf')
    #test regularization strengths
    regs = [0.1, 0.5, 0.1]
    #2layer
    hists = []
    for reg in regs:
        a, b, hist = kfold_it(df, 0.003, reg, linear=False)
        hists.append(hist)
    fig = go.Figure()
    X = list(range(100))
    for i in range(len(hists)):
        fig.add_trace(go.Scatter(x=X, y=np.log(hists[i]),
                        mode='lines',
                        name=str(regs[i]) + ' reg strength'))
    fig.layout.update(
        title='Regularization Strength effect on MSE',
        xaxis_title='Epoch',
        yaxis_title='log(MSE)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.write_image('fig3.pdf')
    #linear
    hists = []
    for reg in regs:
        a, b, hist = kfold_it(df, 0.003, reg, linear=True)
        hists.append(hist)
    fig = go.Figure()
    X = list(range(100))
    for i in range(len(hists)):
        fig.add_trace(go.Scatter(x=X, y=np.log(hists[i]),
                        mode='lines',
                        name=str(regs[i]) + ' reg strength'))
    fig.layout.update(
        title='Regularization Strength effect on MSE',
        xaxis_title='Epoch',
        yaxis_title='log(MSE)',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.write_image('fig4.pdf')



'''
if __name__ == "__main__":
    df = load_df('avocado.csv')
    a, b, hist = kfold_it(df, 0.003, 0.1, linear=False)
    fig = go.Figure()
    X = list(range(100))
    fig.add_trace(go.Scatter(x=X, y=hist,
                    mode='lines',
                    name='0.003 learning rate'))
    fig.write_image('fig.pdf')
'''
