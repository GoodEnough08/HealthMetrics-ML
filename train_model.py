import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

np.random.seed(42)
n = 1500

glucose = np.random.normal(120, 30, n).clip(60, 240)
bmi = np.random.normal(28, 6, n).clip(16, 50)
age = np.random.randint(20, 75, n)
bp = np.random.normal(75, 12, n).clip(50, 130)
insulin = np.random.normal(85, 40, n).clip(15, 300)

# Balanced clinical risk formula incorporating all 5 biometric indicators
risk_score = (
    (glucose > 135) * 0.32 +
    (bmi > 29) * 0.24 +
    (age > 45) * 0.16 +
    (bp > 85) * 0.12 +
    (insulin > 120) * 0.16 +
    np.random.normal(0, 0.07, n)
)
target = (risk_score > 0.42).astype(int)

df = pd.DataFrame({
    'glucose': glucose,
    'bmi': bmi,
    'age': age,
    'bp': bp,
    'insulin': insulin,
    'target': target
})
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
    'precision': round(precision_score(y_test, y_pred) * 100, 2),
    'recall': round(recall_score(y_test, y_pred) * 100, 2),
    'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2)
}

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(list(X.columns), 'features.joblib')
joblib.dump(metrics, 'metrics.joblib')
print("Model trained and artifacts saved successfully!", metrics)