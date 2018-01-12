import pandas as pd

d = pd.read_json('data/origin/test.json')
d.to_hdf('data/origin/test.h5', 'df')