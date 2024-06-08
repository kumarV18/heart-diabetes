
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("Dataset/diabetes.csv")

data.head()

X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train,y_train)
patient_data = {
    "Pregnancies": None,
    "Glucose": None,
    "BloodPressure": None,
    "SkinThickness": None,
    "Insulin": None,
    "BMI": None,
    "DiabetesPedigreeFunction": None,
    "Age": None
}
for feature in patient_data:
  patient_data[feature] = float(input("Enter {} for the patient:".format(feature)))

patient_df = pd.DataFrame(patient_data, index=[0])
prediction = model.predict(patient_df)
print("\nPatient Data:")
print(patient_df)
if prediction[0] == 1:
  print("Diabetes Detected")
else:
  print("Normal")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = KNeighborsClassifier()
model.fit(X_train,y_train)
patient_data = {
    "Pregnancies": None,
    "Glucose": None,
    "BloodPressure": None,
    "SkinThickness": None,
    "Insulin": None,
    "BMI": None,
    "DiabetesPedigreeFunction": None,
    "Age": None
}
for feature in patient_data:
  patient_data[feature] = float(input("Enter {} for the patient:".format(feature)))

patient_df = pd.DataFrame(patient_data, index=[0])
prediction = model.predict(patient_df)
print("\nPatient Data:")
print(patient_df)
if prediction[0] == 1:
  print("Diabetes Detected")
else:
  print("Normal")