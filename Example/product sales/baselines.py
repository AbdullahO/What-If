
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from causallib.estimation import IPW, Standardization, StratifiedStandardization
from causallib.estimation import AIPW, PropensityFeatureStandardization, WeightedStandardization
from causallib.evaluation import evaluate
import pandas as pd
import numpy as np

def ipw_estimator(df, treatment, confounders, outcome, int_labels, method = LogisticRegression, kwargs = {"solver":"saga"}):
    # fit weights
    a = df[treatment]
    y = df[outcome]
    I = len(a.unique())
    learner = method(**kwargs)
    ipw = IPW(learner)
    X = pd.get_dummies(df[confounders])
    
    if len(a.unique()) == 1:
        outcomes =  pd.Series(data = np.ones(I) * y.mean(), index = np.arange(I))
    else:
        ipw.fit(X, a)
        # Estimate average outcome
        outcomes = ipw.estimate_population_outcome(X, a, y)
    ATE = [[outcomes[int_labels[i]] - outcomes[int_labels[0]] for i in range(3)]]
    return ATE

def naive_ATE(df, treatment,  outcome, int_labels):
    # fit weights
    a = df[treatment]
    y = df[outcome]
    I = len(a.unique())
    outcomes =  pd.Series(data = np.ones(I) * y.mean(), index =int_labels)
    for i in int_labels:
        outcomes[i] = y[a == i].mean()
    ATE = [[outcomes[int_labels[i]] - outcomes[int_labels[0]] for i in range(3)]]
    return ATE
