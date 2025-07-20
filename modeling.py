import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import joblib
from typing import Dict, Any
from xgboost import XGBClassifier

# Load preprocessed data
DATA_PATH = 'preprocessed_data.csv'
df = pd.read_csv(DATA_PATH)

print("=== ENHANCED MODELING WITH PIPELINE ===")

# Prepare data for both regression and classification
X_reg = df.drop(['Monthly_Salary', 'Resigned'], axis=1)
y_reg = df['Monthly_Salary']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

X_clf = df.drop(['Resigned', 'Monthly_Salary'], axis=1)
y_clf = df['Resigned']
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train_clf)}")
print(f"Test set size: {len(X_test_clf)}")

# Enhanced Classification Models with Pipeline
print('\n=== CLASSIFICATION MODELS WITH PIPELINE ===')
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

results: Dict[str, float] = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Create pipeline with scaler
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train and predict
    pipe.fit(X_train_clf, y_train_clf)
    y_pred = pipe.predict(X_test_clf)
    
    # Calculate metrics
    acc = accuracy_score(y_test_clf, y_pred)
    results[name] = acc
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_clf, y_pred))
    print("Classification Report:")
    print(classification_report(y_test_clf, y_pred, zero_division='warn'))

# Model Comparison Visualization
print('\n=== MODEL COMPARISON ===')
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracy_values = list(results.values())
plt.bar(model_names, accuracy_values, color='skyblue', alpha=0.7)
plt.ylabel('Accuracy Score')
plt.title('Classification Model Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Model comparison plot saved as 'model_comparison.png'")

# Find and save best model
best_model_name = max(results, key=lambda k: results[k])
best_model = models[best_model_name]
print(f"\n✅ Best Classification Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_classification_model.pkl")
print("✅ Saved best classification model as 'best_classification_model.pkl'")

# Regression Models
print('\n=== REGRESSION MODELS ===')
regressors = {
    'LinearRegression': LinearRegression(),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
}

regression_results: Dict[str, Dict[str, float]] = {}
for name, model in regressors.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)
    regression_results[name] = {'MSE': mse, 'R2': r2}
    print(f'MSE: {mse:.2f}')
    print(f'R2: {r2:.4f}')

# Find best regression model
best_reg_name = max(regression_results.keys(), key=lambda x: regression_results[x]['R2'])
best_reg_model = regressors[best_reg_name]
print(f"\n✅ Best Regression Model: {best_reg_name} with R2 {regression_results[best_reg_name]['R2']:.4f}")

# Save best regression model
joblib.dump(best_reg_model, "best_regression_model.pkl")
print("✅ Saved best regression model as 'best_regression_model.pkl'")

# Print summary
print("\n=== FINAL SUMMARY ===")
print("Classification Results:")
for name, acc in results.items():
    print(f"  {name}: {acc:.4f}")
print(f"\nBest Classification: {best_model_name} ({results[best_model_name]:.4f})")

print("\nRegression Results:")
for name, metrics in regression_results.items():
    print(f"  {name}: R2={metrics['R2']:.4f}, MSE={metrics['MSE']:.2f}")
print(f"\nBest Regression: {best_reg_name} (R2={regression_results[best_reg_name]['R2']:.4f})") 