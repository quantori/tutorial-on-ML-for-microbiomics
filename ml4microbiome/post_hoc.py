from collections import defaultdict 
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

def shap_(
	model: any, 
	X_test: pd.DataFrame, 
) -> shap._explanation.Explanation:
	"""
	Calculate SHAP values.
	
	Args:
		model: trained/tuned model.
		X_test: test feature dataset.
	
	Returns:
		shap_values: shap values calculated by the SHAP package. 
	"""
	explainer = shap.Explainer(model.predict, X_test.to_numpy())
	num_features = X_test.shape[1]
	max_evals = 2 * num_features + 1
	shap_values = explainer(X_test, max_evals=max_evals)
	
	return shap_values

def plot_shap(
	shap_dict: dict,
	X_test_for_shap: list,
	feature_names: list,
) -> plt.Figure:
	"""
	Plot SHAP summary bar and beeswarm plots.
	
	Args:
		shap_dict: keys are indices for reps, values are shap._explanation.Explanation outputs (values, base_values, data) for each sample in each rep. 
		X_test_for_shap: for each rep, the X_test dataset (pandas dataframe) that was used to train the model.
		feature_names: names of features.
	
	Returns:
		The generated matplotlib Figure object.	
	"""
	# This gives us SHAP values for every sample
	SHAP_values_per_fold = []
	for rep in range(len(shap_dict)):
		shap_values = shap_dict[rep].values
		for SHAPs in shap_values:
			SHAP_values_per_fold.append(SHAPs)

	# This gives us feature values for every sample
	test_values_per_fold = []
	
	for rep in range(len(X_test_for_shap)):
		shap_values = X_test_for_shap[rep]
		for i in range(shap_values.shape[0]):
			test_values_per_fold.append(pd.DataFrame(shap_values.iloc[i,:]).T)	

	fig = plt.figure()
	
	ax0 = fig.add_subplot(121)
	shap.summary_plot(np.array(SHAP_values_per_fold), 
					  pd.concat(test_values_per_fold), 
					  [i.split('s__')[1] for i in feature_names], 
					  plot_type='bar', 
					  max_display=15,
					  show=False)
	
	ax1 = fig.add_subplot(122)
	shap.summary_plot(np.array(SHAP_values_per_fold),
					  features=pd.concat(test_values_per_fold),
					  feature_names=[i.split('s__')[1] for i in feature_names],
					  max_display=15,
					  show=False
	)
	plt.gcf().set_size_inches(20,6)
	plt.tight_layout() 
	
	return fig
	
	
def errors(
	yhat: np.ndarray,
	y_test: pd.DataFrame
) -> dict[str, int]:
	"""
	Record sample index of true positives, true negatives, false positives, and false negatives.
	
	Args:
		yhat: model predictions.
		y_test: test target labels.
	
	Returns:
		cm_idx: keys = tp/tn/fp/fn, values = counts.
	"""
	cm_idx = {'tp':[], 'tn':[], 'fp':[], 'fn':[]} 
	for i, idx in enumerate(y_test.index):
		if y_test[idx] == yhat[i] == 1:
			cm_idx['tp'].append(idx)
		elif y_test[idx] == yhat[i] == 0:
			cm_idx['tn'].append(idx)
		elif y_test[idx] == 1 and yhat[i] == 0:
			cm_idx['fn'].append(idx)
		elif y_test[idx] == 0 and yhat[i] == 1:
			cm_idx['fp'].append(idx)
	
	return cm_idx
	
def calc_errs_per_sample(
	results: defaultdict, 
	data_type_key: Literal['metadata_only', 'microbiome_only', 'metadata_microbiome'], 
	tax_level_key: Literal['species', 'genus', 'family', 'all'], 
	alg_key: Literal['random_forest', 'lightGBM', 'logistic_regression_L1'], 
	y_encoded: pd.Series
) -> (defaultdict, defaultdict, defaultdict):
	"""
	For the k-fold CV experiment, calculate the number of errors per sample and number of times each sample was in the test set.
	
	Args:
		results: k-fold CV experimental results. For each replicate per data type, taxonomic level, and ML algorithm, you can access train_index, test_index, model init seed, AUROC, F1, CM, yhat, model.
		data_type_key: data type used for training.
		tax_level_key: taxonomic level used for training.
        alg_key: algorithm used for training.
		y_encoded: y values for full dataset encoded as 0s (no schizophrenia) and 1s (schizophrenia).
	
	Returns:
		errs_per_sample: keys = samples (indexed from 0-n), values = # errors per sample in the k-fold CV experiment.
		occurence: keys = samples (indexed from 0-n), values = # times each sample appeared in the test set in the k-fold CV experiment.
	"""
	y_hats = results[data_type_key][tax_level_key][alg_key]['yhat']
	y_tests = []
	
	occurence = defaultdict(int)
	errs_per_sample = defaultdict(int)
	
	for i in range(len(results[data_type_key][tax_level_key][alg_key]['test_index'])):
		#print(i)
		idxs = results[data_type_key][tax_level_key][alg_key]['test_index'][i]
		y_trues = [y_encoded[i] for i in idxs]
	
		for j in range(len(idxs)):
			sample = idxs[j]
			y_true = y_trues[j]
			y_hat = y_hats[i][j]
			#print("sample",sample)
			occurence[sample] += 1
			if y_true != y_hat:
				errs_per_sample[sample] += 1
			else:
				if sample not in errs_per_sample:
					errs_per_sample[sample] = 0
					
	return errs_per_sample, occurence 	

def plot_err_breakdown(
	df_metadata: pd.DataFrame, 
	errs_per_sample: defaultdict, 
	occurence: defaultdict
) -> plt.Figure:
	"""
	Generate a figure with multiple subplots to visualize the number of errors seen in the k-fold CV experiment for selected demographic characteristics.
    
    Args:
    	df_metadata: metadata mapping sample to demographic factors.
		errs_per_sample: keys = samples (indexed from 0-n), values = # errors per sample in the k-fold CV experiment.
		occurence: keys = samples (indexed from 0-n), values = # times each sample appeared in the test set in the k-fold CV experiment.
    	
    Returns:
    	The generated matplotlib Figure object.
	"""
	to_plot = ['Gender (1:male, 2:female)', 'Marital status', 'Dwelling condition', 'Education level', 'Sample center', 'Diagnosis', 'Smoking', 'Staple food structure', 'Frequency of drinking yogurt or probiotic drinks']
	len(to_plot)
	dfm = df_metadata.copy().reset_index(drop=True)
	dfm['errs'] = [errs_per_sample[i] for i in range(len(dfm))]
	dfm['n_total'] = [occurence[i] for i in range(len(dfm))]
	
	nrows = 4
	ncols = 3
	
	fig, axes = plt.subplots(nrows,ncols, figsize=(10,20))
	feature_idx = 0
	xticks = {}
	for i in range(nrows):
		for j in range(ncols):
			if i >= 3 : continue
			feature_type = to_plot[feature_idx]
			if i > 3: print('BREAK')
			axes[i,j].set_ylim(0,55)
			grouped = dfm.groupby(feature_type).sum()
			y_err = grouped['errs'].values
			y_total = grouped['n_total'].values
			y = 100*(y_err/y_total)
			x = np.arange(len(y_err))
			idx_sort = np.argsort(-y)
			xticks[feature_type] = grouped.index.values[idx_sort]
			y = y[idx_sort]
			axes[i,j].bar(x, y)
			y_total = y_total[idx_sort]
			for x_,y_, y_t in zip(x, y, y_total):
				axes[i,j].text(x_ - 0.1, y_, str(int(y_t/10)))
			axes[i,j].set_xticks(x)
			feature_idx += 1
	
	# Gender
	x = xticks['Gender (1:male, 2:female)']
	remap = {1: 'Male', 2: 'Female'}
	axes[0,0].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[0,0].set_title('Gender')  
	
	# Marital status
	x = xticks['Marital status']
	remap = {'married': 'Married', 'single': 'Single', 'single after divorced': 'Single after\n divorce'}
	axes[0,1].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[0,1].set_title('Marital status')  
	
	# Dwelling condition
	x = xticks['Dwelling condition']
	remap = {'famliy life': 'Family living', 'group living': 'Group living', 'live alone': 'Living alone'}
	axes[0,2].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[0,2].set_title('Dwelling condition')  
	
	# Education level
	x = xticks['Education level']
	remap = {'college': 'College', 'high school': 'High school', 'junior college': 'Junior college',
			 'junior high school': 'Junior high school',
			 'master degree or above': 'Master degree or above', 'primary school': 'Primary school',
			'technical secondary school': 'Technical secondary school', 'uneducated': 'No formal education',
			'vocational high school': 'Vocational high school'}
	axes[1,0].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[1,0].set_title('Education level') 
	
	# Sample center
	x = xticks['Sample center']
	axes[1,1].set_xticklabels(x, rotation=90)
	axes[1,1].set_title('Sample center')  
	
	# Diagnosis
	x = xticks['Diagnosis']
	remap = {'healthy': 'No SCZ', 'first-episode': 'First-episode SCZ', 'relapse': 'Relapse SCZ'}
	axes[1,2].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[1,2].set_title('Diagnosis') 
		 
	# Smoking
	x = xticks['Smoking']
	remap = {'never smoke or occasional smoke ': 'Non-smoker',
			 'intermittent smoking': 'Intermittent smoker',
			 'Smoking less than or equal to 4 cigarettes a day': 'Smoker, light',
			 'smoking 5 to 10 cigarettes a day': 'Smoker, intermediate',
			 'smoking more than 10 cigarettes day or quit smoking failed': 'Smoker, heavy or unable to quit'}
	axes[2,0].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[2,0].set_title('Smoking') 
	
	# Staple food structure
	x = xticks['Staple food structure']
	remap = {'rice': 'Rice', 
			 'rice,flour,coarse food grain': 'Mixed', 
			 'flour': 'Flour', 
			 'coarse food grain': 'Coarse food grain'}
	axes[2,1].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[2,1].set_title('Staple food structure') 
	
	# Frequency of drinking yogurt or probiotic drinks
	x = xticks['Frequency of drinking yogurt or probiotic drinks']
	remap = {'often': 'Medium', 
			 'hardly': 'Low', 
			 'every day': 'High'}
	axes[2,2].set_xticklabels([remap[x_] for x_ in x], rotation=90)
	axes[2,2].set_title('Frequency of drinking yogurt or probiotic drinks') 
	
	# Age
	axes[3,0].scatter(dfm['Age'], dfm['errs'])
	axes[3,0].set_title('Age') 
	axes[3,0].set_xlabel('Age (years)')
	axes[3,0].set_ylabel('Number of errors')
	axes[3,0].set_ylim(-0.5,10.5)
	
	# BMI
	axes[3,1].scatter(dfm['BMI'], dfm['errs'])
	axes[3,1].set_title('BMI') 
	axes[3,1].set_xlabel('BMI')
	axes[3,1].set_ylim(-0.5,10.5)
	
	# Blood pressure
	axes[3,2].scatter(dfm['Systolic pressure(mmHg)']/dfm['Diastolic pressure(mmHg)'], dfm['errs'])
	axes[3,2].set_title('Blood pressure') 
	axes[3,2].set_xlabel('Systolic / diastolic pressure')
	axes[3,2].set_ylim(-0.5,10.5)
	
	for i in range(nrows-1):
		axes[i,0].set_ylabel('Percent errors (%)')
	
	fig.tight_layout()
	
	return fig

def interaction_heatmap(
	var1: str, 
	var2: str, 
	var1_names: np.ndarray, 
	var2_names: np.ndarray, 
	df_metadata: pd.DataFrame, 
	errs_per_sample: defaultdict, 
occurence) -> plt.Figure:
	"""
	Plot a heatmap where each axis is a categorical demographic variable and the value/colour per cell shows the number of errors in the k-fold CV for samples matching categories within each demographic variable.
	
	Args: 
		var1: variable 1 to query / df_metadata column name. 		
		var2: variable 2 to query / df_metadata column name.
		var1_names: categories of variable 1 within corresponding df_metadata column.
		var2_names: categories of variable 2 within corresponding df_metadata column.
    	df_metadata: metadata mapping sample to demographic factors.
		errs_per_sample: keys = samples (indexed from 0-n), values = # errors per sample in the k-fold CV experiment.
		occurence: keys = samples (indexed from 0-n), values = # times each sample appeared in the test set in the k-fold CV experiment.
		
	Returns:
    	The generated matplotlib Figure object.	
	"""
	# Sort education levels from highest to lowest
	ed_levels = ['master degree or above',
			 'college',
			 'junior college',
			 'high school',
			 'technical secondary school',
			 'vocational high school',
			 'junior high school',
			 'primary school',
			 'uneducated']	
			 	
	dfm = df_metadata.copy().reset_index(drop=True)
	dfm['errs'] = [errs_per_sample[i] for i in range(len(dfm))]
	dfm['n_total'] = [occurence[i] for i in range(len(dfm))]
	
	out = []
	for var1_ in var1_names:
		grouped = dfm[dfm[var1] == var1_].groupby(var2).sum()
		x = grouped['errs']/grouped['n_total']*100
		out2 = []
		for var2_ in var2_names:
			try:
				out2.append(round(x[var2_], 1))
			except KeyError:
				out2.append(0)
		out2 = [int(i) if i==0 else (i) for i in out2]
		out.append(out2)
	
	fig, ax = plt.subplots()
	im = ax.imshow(out)
	
	cbar = ax.figure.colorbar(im, ax=ax)
	cbar.ax.set_ylabel('Percent errors (%) per cell', rotation=-90, va='bottom')

	if var1 == 'Education level': var1_names = ed_levels
	if var2 == 'Education level': var2_names = ed_levels
	
	if var1 == 'Gender (1:male, 2:female)': var1_names = ['Female', 'Male']
	if var2 == 'Gender (1:male, 2:female)': var2_names = ['Female', 'Male']
	
	ax.set_xticks(np.arange(len(var2_names)), labels=var2_names, rotation=90)
	ax.set_yticks(np.arange(len(var1_names)), labels=var1_names)
	
	# Loop over data dimensions and create text annotations.
	for i in range(len(var1_names)):
		for j in range(len(var2_names)):
			text = ax.text(j, i, out[i][j],
						   ha='center', va='center', color='w')
	ax.grid(None)		
	
	return fig.tight_layout()		
	
def plot_distrib_errs(errs_per_sample: defaultdict) -> plt.Figure:
	"""
	Plot the distribution of errors per sample across all replicates of the experiment.
	
	Args:
		errs_per_sample: keys = index of sample, values = # errors seen across the experiment.
	
	Returns:
		The generated matplotlib Figure object. 
	"""
	fig, ax = plt.subplots()
	
	counts = defaultdict(int)
	for i in errs_per_sample:
	    counts[errs_per_sample[i]] += 1
	
	ax.bar(counts.keys(), counts.values())
	ax.set_xlim(-0.5,(max(counts.keys())+0.5))
	ax.set_ylim(0, 105)
	
	ax.set_xlabel('Number of errors per sample')
	ax.set_ylabel('Number of samples')
	
	return fig.tight_layout()	   
	
def dataset_wide_err(
	errs_per_sample: defaultdict, 
	occurence: defaultdict
	):
	"""
	Calculates and prints dataset-wide error rate for a k-fold CV experimental condition.
	
	Args:
		errs_per_sample: keys = samples (indexed from 0-n), values = # errors per sample in the k-fold CV experiment.
		occurence: keys = samples (indexed from 0-n), values = # times each sample appeared in the test set in the k-fold CV experiment.
	"""
	err = sum(errs_per_sample.values())/sum(occurence.values())*100
	print('The dataset wide error rate is',str(round(err,2))+'%')