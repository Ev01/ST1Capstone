import pandas as pd

df = pd.read_csv("data.csv")
# Remove any duplicate rows in the dataset
df.drop_duplicates(inplace=True)
#print(df.nunique())
#print(df["bedrooms"].value_counts(normalize=True) * 100)
print(df.describe())