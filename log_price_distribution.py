import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
# Remove any duplicate rows in the dataset
df.drop_duplicates(inplace=True)
df["log_price"].plot(kind="hist")

plt.xlabel("Log Price")

plt.show()