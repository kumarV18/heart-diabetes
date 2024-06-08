
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Dataset/diabetes.csv")

data.head()

glucose_insulin_df = data[["Glucose","Insulin"]]
glucose_insulin_df = glucose_insulin_df.dropna()
correlation = glucose_insulin_df["Glucose"].corr(glucose_insulin_df["Insulin"])
print("Correlation between glucose and insulin:", correlation)
plt.figure(figsize=(8,6))
plt.scatter(glucose_insulin_df["Glucose"],glucose_insulin_df["Insulin"], alpha=0.5)
plt.title("Glucose vs Insulin")
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.grid(True)
plt.show()