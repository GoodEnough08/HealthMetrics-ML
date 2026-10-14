import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 5000

# ─── Generate clinically-realistic correlated patient data ───
# Base biometric distributions with realistic population parameters
age = np.random.normal(48, 16, n).clip(18, 90).astype(int)

# Glucose increases with age, with wider spread for older patients
glucose_base = 80 + (age - 18) * 0.6 + np.random.normal(0, 18, n)
glucose = glucose_base.clip(55, 300)

# BMI follows age-related trends with population variance
bmi_base = 22 + (age - 18) * 0.08 + np.random.normal(0, 4.5, n)
bmi = bmi_base.clip(15, 55)

# Blood pressure correlates with age and BMI
bp_base = 60 + (age - 18) * 0.25 + (bmi - 22) * 0.5 + np.random.normal(0, 9, n)
bp = bp_base.clip(45, 140)

# Insulin correlates with glucose and BMI (insulin resistance pattern)
insulin_base = 30 + (glucose - 80) * 0.4 + (bmi - 22) * 2.5 + np.random.normal(0, 25, n)
insulin = insulin_base.clip(10, 400)

# ─── Clinical risk score using evidence-based thresholds ───
# Continuous risk contributions (not just binary thresholds)
glucose_risk = np.where(glucose < 100, 0,
               np.where(glucose < 126, (glucose - 100) / 26 * 0.3,
               0.3 + (glucose - 126) / 100 * 0.25))

bmi_risk = np.where(bmi < 25, 0,
           np.where(bmi < 30, (bmi - 25) / 5 * 0.2,
           0.2 + (bmi - 30) / 15 * 0.15))

age_risk = np.where(age < 40, 0,
           np.where(age < 55, (age - 40) / 15 * 0.12,
           0.12 + (age - 55) / 35 * 0.08))

bp_risk = np.where(bp < 80, 0,
          np.where(bp < 90, (bp - 80) / 10 * 0.1,
          0.1 + (bp - 90) / 40 * 0.1))

insulin_risk = np.where(insulin < 100, 0,
               np.where(insulin < 150, (insulin - 100) / 50 * 0.15,
               0.15 + (insulin - 150) / 200 * 0.1))

# Interaction terms that reflect real clinical synergy
glucose_bmi_interaction = ((glucose > 126) & (bmi > 30)).astype(float) * 0.08
age_glucose_interaction = ((age > 50) & (glucose > 120)).astype(float) * 0.05

risk_score = (
    glucose_risk +
    bmi_risk +
    age_risk +
    bp_risk +
    insulin_risk +
    glucose_bmi_interaction +
    age_glucose_interaction +
    np.random.normal(0, 0.04, n)  # Small noise
)

# Calibrated threshold for ~35% positive rate (realistic prevalence)
target = (risk_score > 0.28).astype(int)
print(f"Class distribution: {np.bincount(target)} (positive rate: {target.mean():.1%})")

df = pd.DataFrame({
    'glucose': np.round(glucose, 1),
    'bmi': np.round(bmi, 1),
    'age': age,
    'bp': np.round(bp, 1),
    'insulin': np.round(insulin, 1),
    'target': target
})

# ─── Feature Engineering ───
X = df.drop('target', axis=1).copy()
y = df['target']

# Add interaction features the model can leverage
X['glucose_bmi'] = X['glucose'] * X['bmi'] / 1000
X['age_glucose'] = X['age'] * X['glucose'] / 1000
X['bmi_bp'] = X['bmi'] * X['bp'] / 1000
X['insulin_glucose_ratio'] = X['insulin'] / (X['glucose'] + 1)

features = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── Ensemble Model with Calibrated Probabilities ───
gb_clf = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.85,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt',
    random_state=42
)

rf_clf = RandomForestClassifier(
    n_estimators=250,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

lr_clf = LogisticRegression(
    C=0.5,
    max_iter=1000,
    random_state=42
)

# Soft-voting ensemble for smoother probability estimates
ensemble = VotingClassifier(
    estimators=[('gb', gb_clf), ('rf', rf_clf), ('lr', lr_clf)],
    voting='soft',
    weights=[3, 2, 1]
)

# Calibrate probabilities using isotonic regression
model = CalibratedClassifierCV(ensemble, cv=5, method='isotonic')
model.fit(X_train_scaled, y_train)

# ─── Evaluate on held-out test set ───
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Cross-validation for robust metric estimates
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
    'precision': round(precision_score(y_test, y_pred) * 100, 2),
    'recall': round(recall_score(y_test, y_pred) * 100, 2),
    'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
    'f1_score': round(f1_score(y_test, y_pred) * 100, 2),
    'cv_roc_auc_mean': round(cv_scores.mean() * 100, 2),
    'cv_roc_auc_std': round(cv_scores.std() * 100, 2),
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test),
    'n_features': len(features),
    'model_type': 'Calibrated Voting Ensemble (GB + RF + LR)'
}

# ─── Feature importance from the GradientBoosting sub-estimator ───
# Fit GB separately for feature importances
gb_clf.fit(X_train_scaled, y_train)
feature_importances = dict(zip(features, gb_clf.feature_importances_))

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(features, 'features.joblib')
joblib.dump(metrics, 'metrics.joblib')
joblib.dump(feature_importances, 'feature_importances.joblib')

print("\n✅ Model trained and artifacts saved successfully!")
print(f"\n📊 Test Set Metrics:")
for k, v in metrics.items():
    print(f"   {k}: {v}")
print(f"\n🔬 Feature Importances:")
for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"   {feat}: {imp:.4f}")