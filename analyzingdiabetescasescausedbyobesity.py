

import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Dataset/diabetes.csv")

data.head()

bmi_bins = [0,18.5,25,30,35,40,100]
bmi_labels = ["Underweight","Normal","Overweight","Obese Class I","Obese Class II","Obese Class III"]
data["BMI_Category"] = pd.cut(data["BMI"], bins=bmi_bins, labels=bmi_labels)
bmi_diabetes_cases = {}
bmi_total_cases = {}
for label in bmi_labels:
  subset_df = data[data["BMI_Category"]== label]
  bmi_diabetes_cases[label] = len(subset_df[subset_df["Outcome"]== 1])
  bmi_total_cases[label] = len(subset_df)
for label in bmi_labels:
  print("{}:{} diabetes cases out of {} total recorded data".format(label, bmi_diabetes_cases[label], bmi_total_cases[label]))
plt.figure(figsize=(10,6))
plt.bar(bmi_diabetes_cases.keys(),bmi_total_cases.values(),color='skyblue')
plt.title("Total number of cases for each BMI category")
plt.xlabel("BMI Category")
plt.ylabel("Total Number of Cases")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()