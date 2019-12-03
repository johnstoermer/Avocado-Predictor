# Regression Example With Boston Dataset: Baseline
import pandas as pd
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
# load dataset
df = read_csv("avocado.csv", index_col=0)
df = df[['Date', 'Total Volume', 'type', 'year', 'AveragePrice']]
df['Date'] = df['Date'].apply(lambda x: int(x[5:7]))
df['type'] = df['type'].apply(lambda x: 0 if x == 'conventional' else 1)
df = df.rename(columns={'Date': 'month'})
print(df)
dataset = df.values
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
y = dataset[:,4]
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)
estimator.fit(X, y)
prediction = estimator.predict(X)
train_error =  np.abs(y - prediction)
mean_error = np.mean(train_error)
min_error = np.min(train_error)
max_error = np.max(train_error)
std_error = np.std(train_error)
print(mean_error)
print(std_error)
'''
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''
