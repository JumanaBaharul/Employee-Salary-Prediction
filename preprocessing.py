import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load data
DATA_PATH = 'Extended_Employee_Performance_and_Productivity_Data.csv'
df = pd.read_csv(DATA_PATH)

print("=== DATA PREPROCESSING ===")
print(f"Original dataset shape: {df.shape}")

# 1. Check for null values
print("\n=== NULL VALUE ANALYSIS ===")
null_counts = df.isnull().sum()
null_percentages = (null_counts / len(df)) * 100
null_summary = pd.DataFrame({
    'Null Count': null_counts,
    'Null Percentage': null_percentages
})
print(null_summary[null_summary['Null Count'] > 0])

# 2. Handle missing values
print("\n=== HANDLING MISSING VALUES ===")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            # For categorical columns, use mode
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                mode_value = str(mode_values[0])
                df[col].fillna(mode_value, inplace=True)
                print(f"Filled {col} with mode: {mode_value}")
        else:
            # For numerical columns, use median
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled {col} with median: {median_value}")

# 3. Outlier Detection and Handling using IQR
print("\n=== OUTLIER DETECTION & HANDLING ===")
numerical_cols = df.select_dtypes(include=[np.number]).columns
outliers_removed = 0

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    if outlier_count > 0:
        print(f"{col}: {outlier_count} outliers detected")
        # Cap outliers instead of removing
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        outliers_removed += outlier_count

print(f"Total outliers handled: {outliers_removed}")

# 4. Create Box Plots for Outlier Visualization
print("\n=== CREATING BOX PLOTS ===")
num_plots = len(numerical_cols)
cols = 4  # Number of columns in the grid
rows = math.ceil(num_plots / cols)
plt.figure(figsize=(5 * cols, 4 * rows))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(rows, cols, i)
    plt.boxplot(df[col].dropna())
    plt.title(f'{col} Box Plot')
    plt.xticks([])
plt.tight_layout()
plt.savefig('boxplots_outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print("Box plots saved as 'boxplots_outliers.png'")

# 5. Enhanced Label Encoding
print("\n=== LABEL ENCODING ===")
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if col != 'Hire_Date':  # Exclude date columns
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        classes = le.classes_
        if classes is not None:
            print(f"Encoded {col}: {len(classes)} unique values")
        else:
            print(f"Encoded {col}: 0 unique values")

# 6. Date Feature Engineering
print("\n=== DATE FEATURE ENGINEERING ===")
if 'Hire_Date' in df.columns:
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
    df['Hire_Year'] = df['Hire_Date'].dt.year
    df['Hire_Month'] = df['Hire_Date'].dt.month
    df['Hire_Day'] = df['Hire_Date'].dt.day
    df['Years_Since_Hire'] = (pd.Timestamp.now().year - df['Hire_Year'])
    df = df.drop(columns=['Hire_Date'])
    print("Created date features: Hire_Year, Hire_Month, Hire_Day, Years_Since_Hire")

# 7. Feature Scaling
print("\n=== FEATURE SCALING ===")
target_cols = ['Monthly_Salary', 'Resigned']
feature_cols = [col for col in df.columns if col not in target_cols]

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
print(f"Scaled {len(feature_cols)} features")

# 8. Save preprocessed data
PREPROCESSED_PATH = 'preprocessed_data.csv'
df.to_csv(PREPROCESSED_PATH, index=False)

print(f"\n=== PREPROCESSING COMPLETE ===")
print(f"Final dataset shape: {df.shape}")
print(f"Saved to: {PREPROCESSED_PATH}")

# 9. Data Summary
print("\n=== FINAL DATA SUMMARY ===")
print(f"Features: {len(feature_cols)}")
print(f"Target variables: {target_cols}")
print(f"Total samples: {len(df)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB") 