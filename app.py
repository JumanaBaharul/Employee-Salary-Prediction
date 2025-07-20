import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Employee Salary & Resignation Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💼 Employee Salary Analytics & Prediction System</h1>', unsafe_allow_html=True)

# Load models (with error handling)
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load("best_regression_model.pkl")
        clf_model = joblib.load("best_classification_model.pkl")
        return reg_model, clf_model
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run 'python modeling.py' first.")
        return None, None

# Load preprocessed data for reference
@st.cache_data
def load_data():
    try:
        return pd.read_csv('preprocessed_data.csv')
    except FileNotFoundError:
        st.error("❌ Preprocessed data not found! Please run 'python preprocessing.py' first.")
        return None

# Load models and data
reg_model, clf_model = load_models()
df = load_data()

if reg_model is None or clf_model is None or df is None:
    st.stop()

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "🎯 Single Prediction", "📁 Batch Prediction", "📈 Analytics", "ℹ️ About"]
)

if page == "🏠 Home":
    st.markdown("## Welcome to Employee Analytics & Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Overview")
        st.write(f"**Total Employees:** {len(df):,}")
        st.write(f"**Features:** {len(df.columns)}")
        st.write(f"**Target Variables:** Monthly Salary, Resignation Status")
        
        # Basic statistics
        st.markdown("### 📈 Key Statistics")
        salary_stats = df['Monthly_Salary'].describe()
        st.write("**Salary Statistics:**")
        st.write(f"Mean: ₹{salary_stats['mean']:,.2f}")
        st.write(f"Median: ₹{salary_stats['50%']:,.2f}")
        st.write(f"Max: ₹{salary_stats['max']:,.2f}")
    
    with col2:
        st.markdown("### 🎯 Model Performance")
        st.markdown("""
        - **Regression Models:** Predict employee salary
        - **Classification Models:** Predict resignation probability
        - **Best Models:** Automatically selected based on performance
        """)
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Go to **Single Prediction** for individual employee analysis
        2. Use **Batch Prediction** for multiple employees
        3. Explore **Analytics** for data insights
        """)

elif page == "🎯 Single Prediction":
    st.markdown("## 🎯 Single Employee Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Employee Information")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                employee_id = st.number_input("Employee ID", min_value=1, value=1001)
                department = st.selectbox("Department", df['Department'].unique() if 'Department' in df.columns else [0, 1, 2])
                gender = st.selectbox("Gender", df['Gender'].unique() if 'Gender' in df.columns else [0, 1])
                age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
                job_title = st.selectbox("Job Title", df['Job_Title'].unique() if 'Job_Title' in df.columns else [0, 1, 2, 3, 4])
            
            with col2:
                years_at_company = st.slider("Years at Company", float(df['Years_At_Company'].min()), float(df['Years_At_Company'].max()), float(df['Years_At_Company'].mean()))
                education_level = st.selectbox("Education Level", df['Education_Level'].unique() if 'Education_Level' in df.columns else [0, 1, 2, 3])
                performance_score = st.slider("Performance Score", float(df['Performance_Score'].min()), float(df['Performance_Score'].max()), float(df['Performance_Score'].mean()))
                work_hours = st.slider("Work Hours/Week", float(df['Work_Hours_Per_Week'].min()), float(df['Work_Hours_Per_Week'].max()), float(df['Work_Hours_Per_Week'].mean()))
                projects_handled = st.slider("Projects Handled", float(df['Projects_Handled'].min()), float(df['Projects_Handled'].max()), float(df['Projects_Handled'].mean()))
            
            # Additional features
            overtime_hours = st.slider("Overtime Hours", float(df['Overtime_Hours'].min()), float(df['Overtime_Hours'].max()), float(df['Overtime_Hours'].mean()))
            sick_days = st.slider("Sick Days", float(df['Sick_Days'].min()), float(df['Sick_Days'].max()), float(df['Sick_Days'].mean()))
            remote_work = st.slider("Remote Work Frequency", float(df['Remote_Work_Frequency'].min()), float(df['Remote_Work_Frequency'].max()), float(df['Remote_Work_Frequency'].mean()))
            team_size = st.slider("Team Size", float(df['Team_Size'].min()), float(df['Team_Size'].max()), float(df['Team_Size'].mean()))
            training_hours = st.slider("Training Hours", float(df['Training_Hours'].min()), float(df['Training_Hours'].max()), float(df['Training_Hours'].mean()))
            promotions = st.slider("Promotions", float(df['Promotions'].min()), float(df['Promotions'].max()), float(df['Promotions'].mean()))
            satisfaction_score = st.slider("Satisfaction Score", float(df['Employee_Satisfaction_Score'].min()), float(df['Employee_Satisfaction_Score'].max()), float(df['Employee_Satisfaction_Score'].mean()))
            
            submitted = st.form_submit_button("🚀 Predict", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submitted:
            # Create input dataframe
            input_data = {
                'Employee_ID': employee_id,
                'Department': department,
                'Gender': gender,
                'Age': age,
                'Job_Title': job_title,
                'Years_At_Company': years_at_company,
                'Education_Level': education_level,
                'Performance_Score': performance_score,
                'Work_Hours_Per_Week': work_hours,
                'Projects_Handled': projects_handled,
                'Overtime_Hours': overtime_hours,
                'Sick_Days': sick_days,
                'Remote_Work_Frequency': remote_work,
                'Team_Size': team_size,
                'Training_Hours': training_hours,
                'Promotions': promotions,
                'Employee_Satisfaction_Score': satisfaction_score
            }
            
            # Add date features if they exist
            if 'Hire_Year' in df.columns:
                input_data['Hire_Year'] = 2020  # Default year
                input_data['Hire_Month'] = 6
                input_data['Hire_Day'] = 15
                input_data['Years_Since_Hire'] = 4
            
            input_df = pd.DataFrame([input_data])
            
            # Make predictions
            salary_pred = reg_model.predict(input_df)[0]
            resigned_prob = clf_model.predict_proba(input_df)[0][1]
            resigned_pred = clf_model.predict(input_df)[0]
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 💰 Predicted Salary")
            st.markdown(f"**₹{salary_pred:,.2f}**")
            
            st.markdown("### 🚪 Resignation Risk")
            st.markdown(f"**{resigned_prob*100:.1f}%**")
            
            if resigned_pred:
                st.error("⚠️ High resignation risk")
            else:
                st.success("✅ Low resignation risk")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature importance (if available)
            if hasattr(reg_model, 'feature_importances_'):
                st.markdown("### 📊 Top Features")
                feature_importance = pd.DataFrame({
                    'Feature': input_df.columns,
                    'Importance': reg_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(5)
                st.write(feature_importance)

elif page == "📁 Batch Prediction":
    st.markdown("## 📁 Batch Prediction")
    
    st.markdown("### 📤 Upload CSV File")
    st.write("Upload a CSV file with employee data for batch prediction.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="File should contain the same columns as the training data"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_data)} records")
            
            st.markdown("### 📊 Data Preview")
            st.dataframe(batch_data.head())
            
            if st.button("🚀 Run Batch Prediction"):
                with st.spinner("Processing predictions..."):
                    # Make predictions
                    salary_preds = reg_model.predict(batch_data)
                    resigned_probs = clf_model.predict_proba(batch_data)[:, 1]
                    resigned_preds = clf_model.predict(batch_data)
                    
                    # Add predictions to dataframe
                    results_df = batch_data.copy()
                    results_df['Predicted_Salary'] = salary_preds
                    results_df['Resignation_Probability'] = resigned_probs
                    results_df['Resignation_Prediction'] = resigned_preds
                    
                    st.markdown("### 📈 Prediction Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv'
                    )
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Predicted Salary", f"₹{salary_preds.mean():,.2f}")
                    with col2:
                        st.metric("High Risk Employees", f"{sum(resigned_preds)}")
                    with col3:
                        st.metric("Average Resignation Risk", f"{resigned_probs.mean()*100:.1f}%")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

elif page == "📈 Analytics":
    st.markdown("## 📈 Data Analytics")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Salary Distribution")
            fig = px.histogram(df, x='Monthly_Salary', nbins=30, title="Salary Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🚪 Resignation Rate")
            resignation_counts = df['Resigned'].value_counts()
            fig = px.pie(values=resignation_counts.values, names=['Stayed', 'Resigned'], title="Resignation Rate")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔗 Feature Correlations")
        corr_matrix = df.corr()
        fig = px.imshow(corr_matrix, title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(reg_model, 'feature_importances_'):
            st.markdown("### 📊 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': df.drop(['Monthly_Salary', 'Resigned'], axis=1).columns,
                'Importance': reg_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                        title="Top 10 Most Important Features", orientation='h')
            st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This Employee Salary & Resignation Prediction System uses machine learning to:
    - Predict employee monthly salary (Regression)
    - Predict resignation probability (Classification)
    
    ### 🛠️ Technical Stack
    - **Data Processing:** Pandas, NumPy
    - **Machine Learning:** Scikit-learn
    - **Visualization:** Plotly, Matplotlib
    - **Web Framework:** Streamlit
    
    ### 📊 Models Used
    - **Regression Models:** Linear Regression, Random Forest, Gradient Boosting
    - **Classification Models:** Logistic Regression, Random Forest, KNN, SVM, Gradient Boosting
    
    ### 🚀 Features
    - Interactive single employee prediction
    - Batch prediction with CSV upload
    - Real-time analytics and visualizations
    - Model performance comparison
    - Feature importance analysis
    
    ### 📈 Key Metrics
    - **Accuracy Score:** For classification models
    - **R² Score:** For regression models
    - **Mean Squared Error:** For regression models
    - **Confusion Matrix:** For classification models
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Employee Analytics & Prediction System") 