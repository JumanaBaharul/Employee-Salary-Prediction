import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# 1. Histograms for all features
df.hist(figsize=(16, 12))
plt.tight_layout()
plt.savefig('histograms.png')
plt.close()

# 2. Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Feature importance (RandomForest)
# Regression
X_reg = df.drop(['Monthly_Salary', 'Resigned'], axis=1)
y_reg = df['Monthly_Salary']
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_reg, y_reg)
importances_reg = rf_reg.feature_importances_
features_reg = X_reg.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances_reg, y=features_reg)
plt.title('Feature Importance (Regression)')
plt.savefig('feature_importance_reg.png')
plt.close()

# Classification
X_clf = df.drop(['Resigned', 'Monthly_Salary'], axis=1)
y_clf = df['Resigned']
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_clf, y_clf)
importances_clf = rf_clf.feature_importances_
features_clf = X_clf.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances_clf, y=features_clf)
plt.title('Feature Importance (Classification)')
plt.savefig('feature_importance_clf.png')
plt.close()

print('Plots saved: histograms.png, correlation_heatmap.png, feature_importance_reg.png, feature_importance_clf.png') 