import numpy as np
import pandas as pd
from src.diagnostic import diagnostic_test
from src.algorithms import hsvt_ols

# DIAGNOSTIC TESTS
def diagnostic(pre_df, post_df, t=0.99, alpha=0.05):
	columns = ['unit', 'intervention', 'metric']
	
	# sort dataframes 
	pre_df = pre_df.sort_values(by=columns)
	post_df = post_df.sort_values(by=columns)

	# get ivs, units, metrics
	ivs = np.sort(pd.unique(post_df.intervention))
	units = list(np.sort(pd.unique(pre_df.unit)))
	metrics = list(np.sort(pd.unique(pre_df.metric)))

	# get number of units and interventions
	N, K, M = len(units), len(ivs), len(metrics)

	# initialize 
	diagnostic_rslts = np.empty((K*M, 2))
	diagnostic_rslts[:] = np.nan

	# perform diagnostic tests
	for i, iv in enumerate(ivs):
		unit_ids = pd.unique(post_df[post_df.intervention==iv]['unit'])

		for m, metric in enumerate(metrics): 
			diagnostic_rslts[i*M+m, :] = diagnostic_test(pre_df, post_df, unit_ids, metric, iv, t=t, alpha=alpha) 

	# create output dataframe
	df = pd.DataFrame(data=diagnostic_rslts, columns=['pvalues_test', 'energy_statistic'])
	diag_ivs = [ivs[k // M] for k in range(K*M)]
	diag_metrics = metrics * K
	df.insert(0, 'metric', diag_metrics)
	df.insert(0, 'intervention', diag_ivs)
	df['pvalues_test'] = df['pvalues_test'].replace(0, "Fail")
	df['pvalues_test'] = df['pvalues_test'].replace(1, "Pass")
	df['pvalues_test'] = df['pvalues_test'].replace(np.nan, "N/A")
	return df 


# PREDICT COUNTERFACTUALS
def fill_tensor(pre_df, post_df, t=0.99, center=True, rcond=1e-15, alpha=0.05, include_pre=True, rank_method = 'spectral_energy', return_donors_info = False):
	donor_column = ('donor' in pre_df.columns) & ('donor' in post_df.columns)
	columns = ['unit', 'intervention', 'metric']
	if donor_column: 
		columns = ['unit', 'intervention', 'metric', 'donor']
	
	# sort dataframes by (unit, intervention)
	pre_df = pre_df.sort_values(by=columns)
	post_df = post_df.sort_values(by=columns)

	# get all unique interventions (from post-intervention dataframe)
	ivs = np.sort(pd.unique(post_df.intervention))

	# get all units (using pre-intervention data)
	units = list(np.sort(pd.unique(pre_df.unit)))

	# get all metrics
	metrics = list(np.sort(pd.unique(pre_df.metric)))

	# get number of units and interventions
	N, K, M = len(units), len(ivs), len(metrics)
	T0 = pre_df.shape[1]-len(columns)
	T = T0 + post_df.shape[1]-len(columns)

	# check to make sure there aren't any duplicate units in the pre-intervention dataframe
	assert len(pre_df.unit.unique()) == N

	# initialize output dataframe
	yh_data = np.array([])
	idx_data = np.array([])
	donors_dict = {}
	for iv in ivs:
		donors_dict[iv] ={}
		# get potential donors (units who receive intervention 'iv') from POST-intervention data
		unit_ids = pd.unique(post_df[post_df.intervention==iv]['unit'])
		if donor_column:
				donor_list = pd.unique(post_df[post_df.donor==1]['unit'])

		for unit in units: 
			donors_dict[iv][unit] = {}
			# exclude (target) unit from being included in (donor) unit_ids
			donor_units = unit_ids[unit_ids != unit] if unit in unit_ids else unit_ids
			if donor_column:
				 donor_units = donor_units[np.isin(donor_units,donor_list)]
			num_donors = len(donor_units)

			# get pre-intervention target data
			y1 = pre_df.loc[(pre_df.unit==unit)]
			y1 = y1.drop(columns=columns).values.reshape(M*T0)

			# get pre-intervention donor data
			X1 = pre_df.loc[(pre_df.unit.isin(donor_units))]
			donors1 = X1.unit.values	
			X1 = X1.drop(columns=columns).values.reshape(num_donors, M*T0).T 
			for metric in metrics: 
				# get post-intervention donor data
				X2 = post_df.loc[(post_df.unit.isin(donor_units)) & (post_df.intervention==iv) & (post_df.metric==metric)]
				donors = X2.unit.values
				X2 = X2.drop(columns=columns).values.T
				assert np.array_equal(donors,donors1)
				# make counterfactual predictions
				yh, beta = hsvt_ols(X1, X2, y1, t=t, center=center, rcond=rcond, include_pre=include_pre, method = rank_method, return_coefficients = True)
				donors_dict[iv][unit][metric] = dict(zip(donors, beta))
				# append data
				yh_data = np.vstack([yh_data, yh]) if yh_data.size else yh
				idx_data = np.vstack([idx_data, [unit, iv, metric]]) if idx_data.size else np.array([unit, iv, metric])
				#print("unit:%s, metric:%s, intervention: %s, donors: %s"%(unit, metric, i ,donor_units))

	pre_cols = list(pre_df.drop(columns=columns).columns)
	post_cols = list(post_df.drop(columns=columns).columns)
	df_columns = pre_cols + post_cols if include_pre else post_cols  
	df_synth = pd.DataFrame(columns=df_columns, data=yh_data)
	df_synth.insert(0, 'metric', idx_data[:, 2])
	df_synth.insert(0, 'intervention', idx_data[:, 1].astype('int'))
	df_synth.insert(0, 'unit', idx_data[:, 0])
	if return_donors_info: return df_synth, donors_dict
	return df_synth





