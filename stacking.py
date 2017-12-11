#!/usr/bin/python

# Ensembling other submissions
# https://goo.gl/GKJqMo

import os
import numpy as np 
import pandas as pd 
from subprocess import check_output

sub_path = "submissions"
all_files = os.listdir(sub_path)

# load the model with best base performance
sub_base = pd.read_csv('submissions/200_ens_densenet.csv')

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)

concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:6].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:6].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:6].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:6].median(axis=1)

cutoff_lo = 0.8
cutoff_hi = 0.2

# MinMax + BestBase
concat_sub['is_iceberg_base'] = sub_base['is_iceberg']
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
												concat_sub['is_iceberg_max'], 
												np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
															concat_sub['is_iceberg_min'], 
															concat_sub['is_iceberg_base']))

# MinMax + Median
#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
#												concat_sub['is_iceberg_max'], 
#												np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
#															concat_sub['is_iceberg_min'], 
#															concat_sub['is_iceberg_median']))

print(np.sum(concat_sub['is_iceberg'] > 0.99))
print(np.sum(concat_sub['is_iceberg'] < 0.01)) 
print(np.sum(concat_sub['is_iceberg'] == 1))
print(np.sum(concat_sub['is_iceberg'] == 0))

concat_sub['is_iceberg'] = np.where(concat_sub['is_iceberg'] > 0.99, np.ones_like(concat_sub['is_iceberg']), np.where(concat_sub['is_iceberg'] < 0.01, np.zeros_like(concat_sub['is_iceberg']), concat_sub['is_iceberg']))

print(np.sum(concat_sub['is_iceberg'] == 1))
print(np.sum(concat_sub['is_iceberg'] == 0))


concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_median_01.csv', 
													 index=False, float_format='%.6f')