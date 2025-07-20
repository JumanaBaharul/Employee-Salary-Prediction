# Employee Salary Prediction

A machine learning project to predict employee salary and resignation using regression and classification models. Includes data preprocessing, model training, evaluation, visualization, and a Streamlit web app.

## Features
- Data preprocessing (missing values, encoding, scaling)
- Regression and classification (5 models)
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
