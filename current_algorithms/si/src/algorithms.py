import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.matrix import center_data, approximate_rank,approximate_rank2, hsvt
from src.regression import linear_regression, lasso, ridge

# HSVT + OLS 
def hsvt_ols(X1, X2, y1, t=0.99, metric_i = 0,T0 = None, center=True, rcond=1e-15, alpha=0.05, include_pre=True, method = 'spectral_energy', return_coefficients = False, use_lasso = False, use_ridge = False):
	"""
	Input:
		X1: pre-int. donor data (#pre-int samples x #units)
		X2: post-int. donor data (#post-int samples x #units)
		y1: pre-int. target data (#pre-int samples x 1)
		t: level of spectral energy to retain
		center: binary value indicating whether to center pre-int. data

	Output:
		counterfactual estimate of target unit in post-int. regime
	"""
	# if there are no donors, then don't run method 
	if X1.shape[1] == 0:
		if include_pre:
			std = np.zeros(np.concatenate([X1, X2]).shape[0])
			return baseline(np.concatenate([X1, X2])) , np.array([]),std
		else: 
			return baseline(X2), np.array([]), np.zeros(X2.shape[0]),
		#return baseline(np.concatenate([X1, X2])) 
	if T0 == None:
		T0 = X1.shape[0]
	# center training data
	c = 0
	if center: 
		X1, _ = center_data(X1) 
		y1, c = center_data(y1) 

	# ranks
	if method == 'Donoho':
		k1 = approximate_rank2(X1)
		k2 = approximate_rank2(X2)
	else:	
		k1 = approximate_rank(X1, t=t)
		k2 = approximate_rank(X2, t=t)
	# de-noise donor matrices
	X1_hat,u1,s1,v1 = hsvt(X1, rank=k1, return_all = True)
	X2_hat,u2,s2,v2 = hsvt(X2, rank=k2, return_all = True)

	# learn synthetic control via linear regression
	if use_lasso:  beta = lasso(X1_hat, y1)
	elif use_ridge: beta = ridge(X1_hat, y1)
	else: beta = linear_regression(X1_hat, y1, rcond=rcond)

	# forecast counterfactuals
	y2 = X2_hat.dot(beta).T
	yh = np.concatenate([X1_hat[(metric_i)*T0:(metric_i+1)*T0,:].dot(beta).T + c, y2]) if include_pre else y2 

	# estimate sigma
	sigma_1 =  np.mean(np.square(X1_hat-X1))
	sigma_2 =  np.mean(np.square(X2_hat-X2))
	
	# estimate interval width
	S = np.diag(s1**(-2))
	cov_beta = sigma_1*np.dot(np.dot(v1.T,S),v1)
	cov_sq = np.zeros(X2.shape[0])
	for i in range(X2.shape[0]):
		cov_sq[i] =  np.sqrt(1+np.dot(X2[i,:],np.dot(cov_beta,X2[i,:])))
	cov_sq = np.concatenate([np.zeros(T0), cov_sq]) if include_pre else cov_sq
	
	if return_coefficients: return yh, beta, cov_sq
	return yh 

# baseline (simple average)
def baseline(X2):
	return X2.mean(axis=1) 
