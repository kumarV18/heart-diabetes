
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Dataset/heart.csv")

data.head()

correlation_df = data[["Cholesterol","RestingBP"]]
correlation = correlation_df["Cholesterol"].corr(correlation_df["RestingBP"])
print(f"Correlation Between Cholesterol and Blood Pressure: {correlation}")
plt.figure(figsize=(8,6))
plt.scatter(correlation_df["Cholesterol"],correlation_df["RestingBP"], color="blue", alpha=0.5)
plt.title("Correlation between cholesterol and blood pressure")
plt.xlabel("Cholesterol")
plt.ylabel("Blood Pressure")
plt.grid(True)
plt.show()