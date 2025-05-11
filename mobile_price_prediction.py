
# Mobile Phone Price Range Prediction Project

## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## 2. Load Dataset
# Replace 'mobile_price_data.csv' with your actual CSV file path
df = pd.read_csv('mobile_price_data.csv')
df.head()

## 3. Data Exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualizations
sns.countplot(x='price_range', data=df)
plt.title('Price Range Distribution')
plt.show()

sns.boxplot(x='price_range', y='ram', data=df)
plt.title('RAM vs Price Range')
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

## 4. Data Preprocessing
X = df.drop('price_range', axis=1)
y = df['price_range']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

## 6. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## 7. Model Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

## 8. Feature Importance
importances = model.feature_importances_
feature_names = df.drop('price_range', axis=1).columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()

## 9. Save the Model (Optional)
import joblib
joblib.dump(model, 'mobile_price_predictor.pkl')
