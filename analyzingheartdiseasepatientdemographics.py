
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Dataset/heart.csv")

data.head()

age_groups = pd.cut(data["Age"], bins= [0,30,40,50,60,70,80], labels= ["<30","30-40","40-50","50-60","60-70","70-80"])
heart_disease_counts = data[data["HeartDisease"]==1].groupby([age_groups,"Sex"]).size().unstack()
most_common_chest_pain = data.groupby(age_groups)["ChestPainType"].apply(lambda x: x.mode().iloc[0])
average_hr_by_age = data.groupby(age_groups)["MaxHR"].mean()

print("Heart disease cases and most common chest pain type by age group and gender")
print(pd.concat([heart_disease_counts,most_common_chest_pain,average_hr_by_age], axis=1))

heart_disease_counts.plot(kind="bar", stacked=True)
plt.title("Heart Disease Cases by Age Group & Gender")
plt.xlabel("Age Group")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)
plt.legend(title="Gender")
plt.tight_layout()
plt.show()