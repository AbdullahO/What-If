# Data Generation #

This Repo is concerned with geenerating synthetic data.

## Getting started.
 
 A minimal example for generating the data is as follow:

 ```
 from syn_gyn_module import *
# Unit
num_units = 30
# Time
num_timesteps = 30
# Metrics
metric1 = Metric("sales", metric_range = [0,100000] )
metrics = [metric1]
# Interventions:
num_interventions = 3
# initalize and generate
data = SyntheticDataModule(num_units, num_timesteps, num_interventions, metrics)
data.generate()
# subsample
assignment = {"intervention_assignment": "random", "until": num_timesteps,}
data.auto_subsample([assignment])
```
Then you can see the genenrated data through

```
# look at the subsampled dataframe
data.ss_df
# look at full tensor
data.tensor
# look at subsampled tensor
data.ss_tensor

 ```

 See `generate_data.ipynb` for a more complicated example where you can:
 1. Add unit covariates
 2. Add interventions covariates
 3. Add interventions effects based on different subpopulations
 4. Sample the data and assign interventions differentlyy for each subpopulation
 
