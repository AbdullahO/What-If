import numpy as np 
import statsmodels.api as sm
from src.matrix import approximate_rank,approximate_rank2, hsvt

# check if row space of X2 lies within row space of X1 (look at right singular vectors)
def regression_test(v1, v2, alpha=0.05):
	for i in range(v2.shape[1]):
	    model = sm.OLS(v2[:, i], v1)
	    results = model.fit()
	    pvalues = results.pvalues
	    result = True in (pvalue < alpha for pvalue in pvalues)
	    if not result: 
	        return False
	return True 

# check if row space of X2 lies within row space of X1 (look at right singular vectors)
# incomplete...
def energy_test(v1, v2, alpha=0.05):
	P = v1.dot(v1.T)
	delta = v2 - P.dot(v2)
	return np.linalg.norm(delta, 'fro') ** 2 < alpha, np.linalg.norm(delta, 'fro') ** 2 

# diagnostic test 
def diagnostic_test(pre_df, post_df, unit_ids, metric, iv, t=0.99, alpha=0.05): 
	columns = ['unit', 'intervention', 'metric']

	# get dimensions
	N = len(unit_ids)
	M = int(pre_df.loc[pre_df.unit.isin(unit_ids)].shape[0] / N)
	T0 = pre_df.drop(columns=columns).shape[1]
	T1 = post_df.drop(columns=columns).shape[1]

	# pre-int data
	X1 = pre_df.loc[pre_df.unit.isin(unit_ids)]
	X1 = X1.drop(columns=columns).values.reshape(N, M*T0).T 

	# post-int data 
	X2 = post_df.loc[(post_df.unit.isin(unit_ids)) & (post_df.intervention==iv) & (post_df.metric==metric)]
	X2 = X2.drop(columns=columns).values.T 

	# compute row spaces of X1 and X2 (top right singular vectors)
	k1 = approximate_rank(X1, t=t)
	k2 = approximate_rank(X2, t=t)
	u1, s1, v1 = np.linalg.svd(X1, full_matrices=False)
	u2, s2, v2 = np.linalg.svd(X2, full_matrices=False)
	v1 = v1[:k1, :].T
	v2 = v2[:k2, :].T
	s1 = s1[:k1]
	s2 = s2[:k2]
	u1 = u1[:,:k1]
	u2 = u2[:,:k2]
	X1_hat = np.dot(np.dot(u1,np.diag(s1)),v1.T)
	X2_hat = np.dot(np.dot(u2,np.diag(s2)),v2.T)
	
	X1_var = np.square(X1_hat-X1)
	X2_var = np.square(X2_hat-X2)
	# estimate variance
	k1_var = approximate_rank(X1_var, t =t)
	k2_var = approximate_rank(X2_var, t = t)
	
	var1 = hsvt(X1_var, rank=k1_var)
	var2 = hsvt(X2_var, rank=k2_var)
	
	# estimate sigma 
	sigma_1 =  np.mean(np.square(X1_hat-X1))
	sigma_2 =  np.mean(np.square(X2_hat-X2))

	sigma_1_mssA =  np.mean(var1)
	sigma_2_mssA =  np.mean(var2)
	# beta = linear_regression(X1, y1, rcond=rcond)

	# perform regression test
	regression_rslt = regression_test(v1, v2, alpha=alpha)

	# perform energy test 
	inner = (k1*sigma_1+k2*sigma_2)/N+(k1*sigma_1/T0+k2*sigma_2/T1)*(1+np.log(1/alpha)/N)
	t = 8*k2*(inner)
	energy_rslt, energy_value = energy_test(v1, v2, alpha=t)
	return regression_rslt, energy_rslt, energy_value