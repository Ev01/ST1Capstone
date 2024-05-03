from handle_data import get_formatted_dataframe

df = get_formatted_dataframe()

print(df.info())
print(df.describe())