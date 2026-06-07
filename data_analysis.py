import pandas as pd

data = pd.read_csv("dataset/fakenewsnet_combined.csv", encoding="latin1")

print("\nDataset shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nLabel distribution:")
print(data["labels"].value_counts())

print("\nExample article:")
print(data["article_content"].iloc[0])

print("\nArticle length statistics:")
print(data["article_content"].str.len().describe())