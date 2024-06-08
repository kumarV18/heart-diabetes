
import pandas as pd
data = pd.read_csv("Dataset/heart.csv")

data.head()

missing_values = data.isnull().any(axis=1)
print("Rows with missing values:")
print(missing_values)

duplicates_rows = data[data.duplicated()]
print("Duplicate Rows:")
print(duplicates_rows)

data.dropna(axis=0,inplace=True)

data.drop_duplicates(inplace=True)