

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("Dataset/heart.csv")

data.head()

label_encoder = LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["ChestPainType"] = label_encoder.fit_transform(data["ChestPainType"])
data["RestingECG"] = label_encoder.fit_transform(data["RestingECG"])
data["ExerciseAngina"] = label_encoder.fit_transform(data["ExerciseAngina"])
data["ST_Slope"] = label_encoder.fit_transform(data["ST_Slope"])
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X,y)
feature_importances = rf_classifier.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importances': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by="Importances", ascending=False)
print("Ranked list of features by importances:")
print(feature_importances_df)
corr_matrix = X.corr()
importances_matrix = pd.DataFrame(corr_matrix)
importances_matrix['Importance'] = feature_importances
plt.figure(figsize=(12,8))
sns.heatmap(importances_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor="black")
plt.title("Feature Correlation and Importance Heatmap")
plt.show()