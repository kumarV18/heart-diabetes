

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random



data = pd.read_csv("Dataset/heart.csv")

data.head()

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]
categories = {
    'ChestPainType': ['ATA', 'NAP', 'ASY', 'TA'],
    'Sex': ['M', 'F'],
    'RestingECG': ['Normal', 'ST', 'LVH'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Up', 'Flat', 'Down']
}
X = pd.get_dummies(X, columns=categories.keys())
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
random_data = {
    "Age": np.random.randint(20,80),
    "Sex": np.random.choice(categories["Sex"]),
    "ChestPainType": np.random.choice(categories["ChestPainType"]),
    "RestingBP": np.random.randint(100,200),
    "Cholesterol": np.random.randint(100,300),
    "FastingBS": np.random.choice([0,1]),
    "RestingECG": np.random.choice(categories["RestingECG"]),
    "MaxHR": np.random.randint(60,220),
    "ExerciseAngina": np.random.choice(categories["ExerciseAngina"]),
    "Oldpeak": np.random.uniform(0,5),
    "ST_Slope": np.random.choice(categories["ST_Slope"])
}
random_df = pd.DataFrame([random_data])
random_df = pd.get_dummies(random_df, columns=categories.keys())
missing_features = set(X_train.columns) - set(random_df.columns)
for feature in missing_features:
  random_df[feature] = 0
random_df = random_df[X_train.columns]
random_prediction = rf_classifier.predict(random_df)
def print_features(random_data):
  for feature, value in random_data.items():
    print(f"{feature} = {value}")
print("Randomly Generated Patient Data:")
print_features(random_data)
if random_prediction[0] == 1:
  print("Heart Disease Detected")
else:
  print("Normal")