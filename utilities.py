#general ML utilities like reading data, splitting into X and Y, splitting into train/test
#normalizing, standardizing, getting statistics, etc.

import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing as pp
import numpy as np


def read_tmy_csv(fp):

	tmy = pd.read_csv(fp, header=1, low_memory=False)
	tmy_weather = tmy[['ETR (W/m^2)', 'DNI (W/m^2)', 'DHI (W/m^2)', 
									   'TotCld (tenths)', 'Dry-bulb (C)', 'RHum (%)']]
	return tmy_weather


def split_tmy(df, name):

	X_df = df[['ETR (W/m^2)', 'TotCld (tenths)', 'Dry-bulb (C)', 'RHum (%)']]
	Y_df = df[name]
	return (X_df, Y_df)

def train_test(validation_size, seed, X, Y):
	X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, 
																	 test_size=validation_size, random_state=seed)
	return X_train, X_val, Y_train, Y_val

def normalizer(X, Y):
	Y = np.array(Y).reshape(len(Y), 1)

	X_normalizer = pp.MinMaxScaler().fit(X)
	Y_normalizer = pp.MinMaxScaler().fit(Y)
	#reshape Y because it's 1d, this is a requirement for this sklearn module
	X_scaled = X_normalizer.transform(X)
	Y_scaled = Y_normalizer.transform(Y)
	return X_scaled, Y_scaled.ravel(), X_normalizer, Y_normalizer
	
def standardizer(X, Y):
	#NOTE, reshape is happening right away because we're fitting first now
	Y = np.array(Y).reshape(len(Y), 1)
		
	X_standardizer = pp.StandardScaler().fit(X)
	Y_standardizer = pp.StandardScaler().fit(Y)
	#reshape Y because it's 1d, this is a requirement for this sklearn module
	#NOTE, we return the raveled Y, which brings it back to it's original dimensions

	X_scaled = X_standardizer.transform(X)
	Y_scaled = Y_standardizer.transform(Y)
	return X_scaled, Y_scaled.ravel(), X_standardizer, Y_standardizer










