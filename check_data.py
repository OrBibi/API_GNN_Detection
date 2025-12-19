import pandas as pd

df = pd.read_parquet('data/processed/splits/dataset_1_test.parquet')
print('First request:', df.iloc[0]['request'])
print('First response:', df.iloc[0]['response'])
print('Label:', df.iloc[0]['label'])