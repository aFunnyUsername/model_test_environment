import utilities as util
import pandas as pd
from sklearn.linear_model import LassoLars
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib


tmy_df = util.read_tmy_csv('tmy3MSPairport.csv')
tmy_df_remove_zero = tmy_df[tmy_df['ETR (W/m^2)'] != 0]

#data prep will take in the dataframe we're interested in, a seed value, validation size and dependent variable
#in this case, DNI or DHI
def data_prep(df, seed, validation_size, dep_var):
	X_df, Y_df = util.split_tmy(df, dep_var)

	#now, we have the data, let's look at it without the zeros removed, 
	#and train the LL model on that

	X = X_df.values
	Y = Y_df.values

	#NOTE I think this is just for randomization
	X_train, X_val, Y_train, Y_val = util.train_test(validation_size, seed, X, Y)
	#NOTE, in the past, I've been overwriting my X_train when I run the randomizer again.  
	#so the training data for the two Y-values wasn't actually different
	#now we scale the data, we also keep the scaler used so we can invert it later
	X_train_scaled, Y_train_scaled, X_norm, Y_norm = util.standardizer(X_train, Y_train)
	return (X_train_scaled, Y_train_scaled, X_val, Y_val, X_norm, Y_norm)

#model = ('Lasso Lars', LassoLars())
model = ('Support Vector Regression', SVR())
DNI_seed = 7
DHI_seed = 5
validation_size = 0.20
kfolds = 10
evaluation_metrics = ['Mean Absolute Error', 'R Squared']
stats = ['Mean', 'Stdev']
#evaulation_metrics = ['neg_mean_absolute_error', 'r2']
#NOTE, these are the strings required for the scoring parameter in the cross_val_score method from sklearn

#NOTE, data packages will have all the returned values from data_prep(), according to the
#input parameters that were passed in.
DNI_data_package = data_prep(tmy_df_remove_zero, DNI_seed, validation_size, 'DNI (W/m^2)')
DHI_data_package = data_prep(tmy_df_remove_zero, DHI_seed, validation_size, 'DHI (W/m^2)')




#trainer function will take in the necessary parts from the data package and output relevant summary stats
#for the models we're interested in

def trainer(X, Y, seed, k, model_tuple, metrics):
	name = model_tuple[0]
	model = model_tuple[1]

	results_mean_list = []
	results_std_list = []

	kfold = model_selection.KFold(n_splits=k, random_state=seed)
	for metric in metrics:
		if metric == 'Mean Absolute Error':
			scoring = 'neg_mean_absolute_error'
		elif metric == 'R Squared':
			scoring = 'r2'
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		results_mean_list.append(cv_results.mean())
		results_std_list.append(cv_results.std())
	
	return (results_mean_list, results_std_list)

DNI_results = trainer(DNI_data_package[0], DNI_data_package[1], DNI_seed, kfolds, model, evaluation_metrics)
DHI_results = trainer(DHI_data_package[0], DHI_data_package[1], DHI_seed, kfolds, model, evaluation_metrics)

def results_df_maker(metrics, stats, means, stds):
	results_df = pd.DataFrame()
	results_df['Stats'] = stats
	results_df.set_index('Stats')	
	for i, metric in enumerate(metrics):
		stats_list = []
		stats_list.append(means[i])
		stats_list.append(stds[i])
		results_df[metric] = stats_list
	return results_df

DNI_df = results_df_maker(evaluation_metrics, stats, DNI_results[0], DNI_results[1])
DHI_df = results_df_maker(evaluation_metrics, stats, DHI_results[0], DHI_results[1])
print('DNI results for ' + model[0] + ': ')
print(DNI_df)
print('\n')
print('DHI results for ' + model[0] + ': ')
print(DHI_df)












