# Employee Salary Prediction

A machine learning project to predict employee salary and resignation using regression and classification models. Includes data preprocessing, model training, evaluation, visualization, and a Streamlit web app.

## Features
- Data preprocessing (missing values, encoding, scaling)
- Regression and classification (5+ models)
- Model evaluation (metrics, confusion matrix, R2, MSE, accuracy)
- Data visualization (histograms, heatmaps, feature importance)
- Streamlit web app for predictions

## Setup
1. Clone the repo and place your dataset as `Extended_Employee_Performance_and_Productivity_Data.csv` in the root folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run preprocessing:
   ```bash
   python preprocessing.py
   ```
4. Train and evaluate models:
   ```bash
   python modeling.py
   ```
5. Generate visualizations:
   ```bash
   python visualization.py
   ```
6. Launch the web app:
   ```bash
   streamlit run app.py
   ```

## Web App with ngrok
To share your app online, install ngrok and run:
```bash
ngrok http 8501
```
Share the generated public URL.

## Project Structure
- `preprocessing.py`: Data cleaning and feature engineering
- `modeling.py`: Model training and evaluation
- `visualization.py`: EDA and feature importance plots
- `app.py`: Streamlit web app
- `requirements.txt`: Python dependencies
- `README.md`: Project documentation

## Results & Insights

### Preprocessing
- **Dataset size:** 100,000 samples, 20 → 23 columns after feature engineering  
- **Null values:** None detected  
- **Encoding:** Department (9 values), Gender (3), Job Title (7), Education Level (4)  
- **Feature engineering:** Added Hire Year, Hire Month, Hire Day, and Years Since Hire  
- **Scaling:** 21 features standardized  


### Classification (Predicting *Resigned*)
- **Models tested:** Logistic Regression, Random Forest, Gradient Boosting, KNN, XGBoost  

**Results:**  
- Logistic Regression: **90.14% accuracy**  
- Random Forest: **90.14% accuracy**  
- Gradient Boosting: **90.14% accuracy**  
- KNN: **89.43% accuracy**  
- XGBoost: **90.09% accuracy**  

**Observation:**  
Accuracy is consistent across models, but they mostly predict *“not resigned.”*  
Precision/recall for the minority class (resigned employees) was very low due to class imbalance.  


### Regression (Predicting *Monthly Salary*)
- **Models tested:** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor  

**Results:**  
- Linear Regression: **R² = 0.29**, MSE ≈ **1.33M** → weak linear fit  
- Random Forest Regressor: **R² = 1.0**, MSE = **0.0** → indicates possible overfitting or leakage  
- Gradient Boosting Regressor: **R² = 1.0**, MSE ≈ **77.4** → similarly suspiciously perfect  

**Observation:**  
Tree-based models gave near-perfect results (likely overfitting or data leakage).  
Linear Regression gave more realistic but weaker results.  
