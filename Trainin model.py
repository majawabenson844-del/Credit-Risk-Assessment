import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import joblib
import os 
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
path = "combined_solar_dataset.csv"
df = pd.read_csv(path)

# 2. Rename columns to match your dataset
df.columns = [
    'Gender',
    'Age',
    'Marital_Status',
    'Employment',
    'Residence',
    'Home_Ownership',
    'Number_Dependents',
    'Loan_Amount',
    'Decision'
]
print(df["Decision"].unique())
print(df["Decision"].dtype)
# 3. Target
y = df["Decision"].astype(int)

# 4. Separate predictors
X = df.drop("Decision", axis=1)

# Split categorical vs continuous
categorical_cols = ['Gender','Marital_Status','Employment','Residence','Home_Ownership']
continuous_cols = ['Age','Number_Dependents','Loan_Amount']

# Encode categorical only
encoder = OrdinalEncoder()
X_categorical = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=categorical_cols)

# Keep continuous as is
X_continuous = X[continuous_cols].astype(float)

# Combine back
X_encoded = pd.concat([X_categorical, X_continuous], axis=1)

# ✅ Use all variables
important_features = X_encoded.columns.tolist()
print("Selected predictors (all variables):", important_features)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scale (continuous + encoded categorical together)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train SVM
svm_model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, class_weight="balanced")
svm_model.fit(X_train_scaled, y_train)

# Evaluate SVM
y_pred = svm_model.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 8. Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf_model.fit(X_train_scaled, y_train)

# 9. Voting Ensemble
ensemble = VotingClassifier(
    estimators=[("svm", svm_model), ("rf", rf_model)],
    voting="soft"
)
ensemble.fit(X_train_scaled, y_train)

# Evaluate Ensemble
y_pred_ensemble = ensemble.predict(X_test_scaled)
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print(classification_report(y_test, y_pred_ensemble))
print(confusion_matrix(y_test, y_pred_ensemble))

# 10. Stacking Ensemble
stacking = StackingClassifier(
    estimators=[("svm", svm_model), ("rf", rf_model)],
    final_estimator=LogisticRegression(),
    passthrough=True
)
stacking.fit(X_train_scaled, y_train)

# Evaluate Stacking
y_pred_stack = stacking.predict(X_test_scaled)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))
print(classification_report(y_test, y_pred_stack))
print(confusion_matrix(y_test, y_pred_stack))

# 11. Save artifacts
save_dir = r"C:\Users\ACCENTURE\OneDrive\Desktop\system project"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(svm_model, os.path.join(save_dir, "svm_model.pkl"))
joblib.dump(rf_model, os.path.join(save_dir, "rf_model.pkl"))
joblib.dump(ensemble, os.path.join(save_dir, "ensemble_model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
joblib.dump(encoder, os.path.join(save_dir, "encoder.pkl"))
joblib.dump(important_features, os.path.join(save_dir, "selected_features.pkl"))
