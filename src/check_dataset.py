import pandas as pd

file_path = "data/Benign-Monday-no-metadata.parquet"

df = pd.read_parquet(file_path)

print("Veri seti boyutu:")
print(df.shape)

print("\nSütunlar:")
print(df.columns.tolist())

print("\nİlk 5 satır:")
print(df.head())

print("\nVeri tipleri:")
print(df.dtypes)

print("\nEksik değerler:")
print(df.isnull().sum())

print("\nLabel dağılımı:")
print(df["Label"].value_counts())