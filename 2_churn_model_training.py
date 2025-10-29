import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("🚀 Starting Customer Churn Model Training...\n")

print("📊 Loading data from database...")
conn = sqlite3.connect('database/churn_analysis.db')
df = pd.read_sql_query("SELECT * FROM customers", conn)
conn.close()

print(f"✅ Loaded {len(df)} records")
print(f"Original shape: {df.shape}")

print("\n🧹 Cleaning data...")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

original_len = len(df)
df = df.dropna()
print(f"- Removed {original_len - len(df)} rows with missing values")

df = df.drop('customerID', axis=1)

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

print(f"✅ Clean dataset shape: {df.shape}")

print("\n🔧 Engineering features...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"- Numeric columns: {len(numeric_cols)}")
print(f"- Categorical columns: {len(categorical_cols)}")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"✅ Encoded dataset shape: {df_encoded.shape}")

feature_columns = [col for col in df_encoded.columns if col != 'Churn']
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("\n📊 Splitting data...")

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"- Training set: {X_train.shape}")
print(f"- Test set: {X_test.shape}")
print(
    f"- Churn in training: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
print(f"- Churn in test: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")

print("\n⚖️ Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Scaler saved")

print("\n🌲 Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Random Forest Results:")
print(classification_report(y_test, rf_pred))
print(f"🎯 ROC-AUC Score: {roc_auc_score(y_test, rf_proba):.4f}")

print("\n🚀 Training XGBoost...")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 XGBoost Results:")
print(classification_report(y_test, xgb_pred))
print(f"🎯 ROC-AUC Score: {roc_auc_score(y_test, xgb_proba):.4f}")

print("\n⭐ Calculating feature importance...")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Top 10 Important Features ---")
print(feature_importance.head(10))

feature_importance.to_csv('outputs/churn_feature_importance.csv', index=False)
print("✅ Feature importance saved")

print("\n📈 Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d',
            cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Random Forest Confusion Matrix',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)

sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d',
            cmap='Greens', ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title('XGBoost Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/churn_confusion_matrices.png',
            dpi=300, bbox_inches='tight')
print("✅ Confusion matrices saved")

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_proba):.3f})',
         linewidth=2, color='blue')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, xgb_proba):.3f})',
         linewidth=2, color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Customer Churn Prediction',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig('outputs/churn_roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ ROC curve saved")

print("\n💾 Exporting results...")

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted_RF': rf_pred,
    'Predicted_XGB': xgb_pred,
    'Probability_RF': rf_proba,
    'Probability_XGB': xgb_proba
})
results_df.to_csv('outputs/churn_predictions.csv', index=False)
print("✅ Predictions saved for Power BI/Tableau")

print("\n💾 Saving models...")

with open('models/rf_churn_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('models/xgb_churn_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("✅ Models saved successfully!")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"\n📁 Files created in outputs/:")
print("   - churn_predictions.csv")
print("   - churn_feature_importance.csv")
print("   - churn_confusion_matrices.png")
print("   - churn_roc_curve.png")
print(f"\n🤖 Models saved in models/:")
print("   - rf_churn_model.pkl")
print("   - xgb_churn_model.pkl")
print("   - scaler.pkl")
print("   - feature_columns.pkl")
print("\n📌 Next step: Run Streamlit dashboard")
print("   Command: streamlit run 3_churn_dashboard.py")
print("="*60)
