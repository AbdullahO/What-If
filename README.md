# What If
What If Engine


This package integrates ideas from different research papers such as:

1. **Causal Matrix Completion**, by Anish Agarwal, Munther Dahleh, Devavrat Shah, Dennis Shen. ([link](https://proceedings.mlr.press/v195/agarwal23c/agarwal23c.pdf)).
2. **On multivariate singular spectrum analysis and its variants**, by Anish Agarwal, Abdullah Alomar, and Devavrat Shah. ([link](https://arxiv.org/abs/2006.13448)).
3. **Change Point Detection via Multivariate Singular Spectrum Analysis**, by Arwa Alanqary, Abdullah Alomar, Devavrat Shah. ([link](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c348616cd8a86ee661c7c98800678fad-Abstract.html)).
4. **Synthetic A/B Testing using Synthetic Interventions**,  by Anish Agarwal, Devavrat Shah, Dennis Shen. ([link](https://arxiv.org/abs/2006.07691)).
5. **SAMoSSA: Multivariate Singular Spectrum Analysis with Stochastic Autoregressive Noise**, by Abdullah Alomar, Munther Dahleh, Sean Mann, Devavrat Shah. ([link](https://arxiv.org/pdf/2305.16491.pdf))


## Getting started

Assume access to a dataframe of the form 

| datetime | unit | action     | metric | covariate |
| -------- | ---- | ---------- | ------ | --------- |
| 1/1/2020 | 1    | "action 0" | 0.1    | 0         |
| 2/1/2020 | 1    | "action 1" | 2      | 1         |
| 1/1/2020 | 2    | "action 0" | 4      | 1         |

where $N$ units are observed in $T$ timesteps under $D$ different actions and interventions. The What-if tool allows you to answer counterfactual questions about any unit at any time (in the past) under any intervention. Further, it allows you to forecast your metric for any unit under any intervention. 

To get started, import and fit the model on the dataframe as follows

```python
from whatIf.algorithms.snn import SNN
model = SNN(verbose=False, L = 4, k_factors=8, num_lags_forecasting= 60)


model.fit(
          # first input is the dataframe
          df= df, 
          # for unit_column choose the column name in df with the unique identifier for units (unit in this data)
          unit_column="unit",
          # for time_column choose the column name in df with the timestamps (datetime in this data)
          time_column="datetime",
          # for metrics choose the column name in df with the metric measurements (metric1 in this data)
          metrics = ["metric1"],
          # for actions choose the column(s) name in df indicating the intervention (action in this data)
          actions = ["action"]
)

```


Then use `query` to answer counterfactual questions,

```python
df_query = model.query(
            # which unit(s) to query?
            units= [0], 
            # time frame to query?
            time = ["2020-01-20", "2020-02-20"],
            # metric and action to query
            metric= 'sales', action= 'action 0', 
            # time range at which the selected action should be assigned (for other timesteps,  the observed action will be assigned)
            action_time_range=["2020-01-20", "2020-02-20"])
```


And use `forecast` to forecast!

```python
# forecat 10 steps ahead
forecast = model.forecast(
            units= [0], 
            steps_ahead = 10,
            metric= 'sales', action= 'action 0', 
            )
```

## Examples

See the product sales example ([here](https://github.com/AbdullahO/whatIf/blob/main/Example/product%20sales/product%20sales%20example.ipynb)).
