# ==========================================
# Import Libraries
# ==========================================

import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# Step 1 : Load Dataset
# ==========================================

df = pd.read_csv("food_delivery.csv")

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(df.head())

# ==========================================
# Step 2 : Data Cleaning
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

df = df.drop_duplicates()

# ==========================================
# Step 3 : Encode Categorical Columns
# ==========================================

encoder = LabelEncoder()

df["Weather"] = encoder.fit_transform(df["Weather"])
df["Traffic"] = encoder.fit_transform(df["Traffic"])
df["Vehicle"] = encoder.fit_transform(df["Vehicle"])
df["Delivery_Status"] = encoder.fit_transform(df["Delivery_Status"])

# ==========================================
# Remove Delivery ID
# ==========================================

if "Delivery_ID" in df.columns:
    df = df.drop("Delivery_ID", axis=1)

# ==========================================
# Step 4 : Features and Target
# ==========================================

X = df.drop("Delivery_Status", axis=1)
y = df["Delivery_Status"]

# ==========================================
# Step 5 : Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data :", X_train.shape)
print("Testing Data :", X_test.shape)

# ==========================================
# Step 6 : Train Model
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==========================================
# Step 7 : Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 8 : Model Evaluation
# ==========================================

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# Step 9 : Save Model
# ==========================================

pickle.dump(model, open("model.pkl", "wb"))

print("\nModel Saved Successfully")
print("File Name : model.pkl")