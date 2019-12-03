# Regression Example With Boston Dataset: Standardized
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('avocado.csv', index_col=0)
df = df[['Date', 'Total Volume', 'type', 'year', 'AveragePrice']]
df['Date'] = df['Date'].apply(lambda x: int(x[5:7]))
df['type'] = df['type'].apply(lambda x: 0 if x == 'conventional' else 1)
df = df.rename(columns={'Date': 'month'})
print(df)
# split into input (X) and output (Y) variables
X = df.values[:,0:4]
Y = df.values[:,4]
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
	return model
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
