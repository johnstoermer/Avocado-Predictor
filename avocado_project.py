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

def load_df(f):
    df = pd.read_csv(f, index_col=0)
    df = df[['AveragePrice', 'Date', 'Total Volume', 'type', 'year', 'region']]
    df['Date'] = df['Date'].apply(lambda x: int(x[5:7]))
    df['type'] = df['type'].apply(lambda x: 0 if x == 'conventional' else 1)
    df = df.rename(columns={'Date': 'month'})
    df = pd.get_dummies(df)
    df = df.sample(frac=1)
    return df

if __name__ == "__main__":

    df = load_df('avocado.csv')

    print(df)
    print(len(df.columns))

    Xtrain = df.values[:,1:]
    ytrain = df.values[:,0]

    avocado_model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(58, input_dim=58, activation='relu'),
      tf.keras.layers.Dense(58, input_dim=58, activation='relu'),
      tf.keras.layers.Dense(1)
    ])

    avocado_model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

    avocado_model.fit(Xtrain, ytrain, epochs=100)

    prediction = avocado_model.predict(Xtrain)

    train_error =  (ytrain - prediction) ** 2
    MSE = np.mean(train_error)
