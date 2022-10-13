
# Dataset sample

This data is synthetically generated according to the tensor factor model. There are two files: 
## 1. store_sales_simple.csv
This table contains the observed data across units (herein, stores) and time. The columns are:
**1. unit_id**
**2. time**
**3. location:** location of the store, either LA, NEW YORK or Boston
**4. size: **size of the store, either large, medium or small. 
**5. sales:** metric of interest, the daily sales in the store.
**6. ads:** intervention columns; either ad_0 (control), ad_1, or ad_2. 

The data is generated such that ad_1 is very effective across all units, while ad_2 is as good as ad_0. 


## 2. tensor_store_sales_simple.csv

this file contains the true tensor. It is represneted by the format (coordinates, value) where the first four columns:

1.`unit`
2.`time_index`
3.`action`
4.`metric`

Represents the coordinates, while value column represent the metric value at these coordinates. 
 - Note that the timestamp corresponsing to the time index is given in the column `time`. 
 - The actions coordinates 0,1, and 2 correspond to `ad_0`, `add_1` and `add_2` respectively. 
 
