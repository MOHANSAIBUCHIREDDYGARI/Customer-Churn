# app.py - UPDATED WITH PREPROCESSING
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import os

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# ============================================================================
# PREPROCESSING FUNCTIONS (SAME AS YOUR NOTEBOOK)
# ============================================================================
def preprocess_input_data(df):
    """Preprocess raw customer data exactly like training data"""
    df_clean = df.copy()
    
    # 1. Handle TotalCharges if present
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
    
    # 2. Encode binary variables
    binary_mappings = {
        'gender': {'Female': 0, 'Male': 1, 'female': 0, 'male': 1, 'F': 0, 'M': 1},
        'Partner': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0},
        'Dependents': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0},
        'PhoneService': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0},
        'SeniorCitizen': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0, 1: 1, 0: 0}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping)
            # Fill any NaN with 0 (assume "No")
            df_clean[col].fillna(0, inplace=True)
            # Convert to int
            df_clean[col] = df_clean[col].astype(int)
    
    # 3. Handle service columns
    service_map = {'No': 0, 'Yes': 1, 'No phone service': 0, 'No internet service': 0,
                   'no': 0, 'yes': 1, 'No Phone Service': 0, 'No Internet Service': 0}
    
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in service_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(service_map)
            df_clean[col].fillna(0, inplace=True)
            df_clean[col] = df_clean[col].astype(int)
    
    # 4. One-hot encode categorical columns
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    
    for cat_col in cat_cols:
        if cat_col in df_clean.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df_clean[cat_col], prefix=cat_col, drop_first=True)
            
            # Add to dataframe
            df_clean = pd.concat([df_clean, dummies], axis=1)
            
            # Drop original column
            df_clean = df_clean.drop(cat_col, axis=1)
    
    return df_clean

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    try:
        with open("churn_system.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

churn_system = load_model()

# Initialize session state
if "predictions_df" not in st.session_state:
    st.session_state["predictions_df"] = None

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "📍 Navigation",
        ["Home", "Upload & Predict", "Risk Dashboard", "High Risk", "Download Results", "Data Guide"],
        icons=["house", "upload", "bar-chart", "exclamation-triangle", "download", "info-circle"],
        default_index=0
    )
    
    if churn_system:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model Not Loaded")

# Home Page
if selected == "Home":
    st.title("📊 Customer Churn Prediction System")
    
    if churn_system:
        st.success("✅ System Ready for Predictions!")
        
        # Show expected features
        with st.expander("🔍 Expected Features After Preprocessing"):
            st.write(f"**{len(churn_system.feature_names)} features:**")
            st.write(churn_system.feature_names[:10])
            if len(churn_system.feature_names) > 10:
                st.write(f"... and {len(churn_system.feature_names) - 10} more")
        
        # Download sample CSV with RAW data (not preprocessed)
        st.write("### Need Sample Data?")
        st.write("Download this sample CSV with **raw categorical data** (it will be preprocessed automatically):")
        
        # Create sample raw data
        sample_raw_data = pd.DataFrame({
            'customerID': ['001', '002', '003', '004', '005'],
            'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
            'SeniorCitizen': [0, 1, 0, 0, 1],
            'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
            'Dependents': ['No', 'Yes', 'No', 'No', 'Yes'],
            'tenure': [12, 24, 1, 36, 6],
            'PhoneService': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No', 'Yes', 'No'],
            'OnlineBackup': ['Yes', 'No', 'No', 'Yes', 'Yes'],
            'DeviceProtection': ['No', 'Yes', 'No', 'No', 'Yes'],
            'TechSupport': ['No', 'Yes', 'No', 'No', 'Yes'],
            'StreamingTV': ['No', 'Yes', 'No', 'Yes', 'No'],
            'StreamingMovies': ['No', 'Yes', 'No', 'Yes', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check'],
            'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
            'TotalCharges': [358.20, 1366.80, 53.85, 1522.80, 424.20]
        })
        
        csv = sample_raw_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Sample CSV (Raw Data)",
            data=csv,
            file_name="sample_customers_raw.csv",
            mime="text/csv"
        )
        
        st.write("**Note:** The system will automatically preprocess this data (convert 'Female' to 0, 'Yes' to 1, etc.)")

# Upload & Predict Page
if selected == "Upload & Predict":
    st.title("📤 Upload & Predict")
    
    if not churn_system:
        st.error("Model not loaded. Please check the Home page.")
    else:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=['csv'])
        
        if uploaded_file:
            try:
                # Read the CSV
                customer_df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded! {len(customer_df)} customers")
                
                # Show raw data
                st.write("### Raw Data Preview")
                st.dataframe(customer_df.head())
                
                # Preprocess the data
                with st.spinner("Preprocessing data..."):
                    processed_df = preprocess_input_data(customer_df)
                    
                    # Show preprocessing results
                    with st.expander("🔧 Show Preprocessed Data"):
                        st.write(f"**Before:** {customer_df.shape[1]} columns")
                        st.write(f"**After:** {processed_df.shape[1]} columns")
                        st.dataframe(processed_df.head())
                
                # Check for required features
                missing_features = [f for f in churn_system.feature_names if f not in processed_df.columns]
                
                if missing_features:
                    st.warning(f"⚠️ Missing {len(missing_features)} features after preprocessing")
                    with st.expander("Show missing features"):
                        st.write(missing_features[:10])
                    st.info("Missing features will be filled with zeros.")
                
                # Make predictions
                if st.button("🔮 Make Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        predictions_df = churn_system.batch_predict(processed_df)
                        st.session_state["predictions_df"] = predictions_df
                    
                    st.success("✅ Predictions Complete!")
                    
                    # Show summary
                    st.write("### Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", len(predictions_df))
                    col2.metric("Will Churn", predictions_df['will_churn'].sum())
                    high_risk = len(predictions_df[predictions_df['risk_level'] == 'HIGH'])
                    col3.metric("High Risk", high_risk)
                    col4.metric("Churn Rate", f"{(predictions_df['will_churn'].sum()/len(predictions_df)*100):.1f}%")
                    
                    # Show predictions
                    st.write("### Predictions Preview")
                    st.dataframe(predictions_df.head())
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure your CSV has the same structure as the training data.")

# Data Guide Page
if selected == "Data Guide":
    st.title("📋 Data Format Guide")
    
    st.write("""
    ### Required Data Format
    
    Your CSV should have columns similar to the original dataset. Here are the expected columns:
    """)
    
    data_format = pd.DataFrame({
        'Column Name': ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'],
        'Expected Values': [
            'Any string (will be dropped)',
            '"Female" or "Male"',
            '0 or 1',
            '"Yes" or "No"',
            '"Yes" or "No"',
            'Number (1-72)',
            '"Yes" or "No"',
            '"Yes", "No", or "No phone service"',
            '"DSL", "Fiber optic", or "No"',
            '"Yes", "No", or "No internet service"',
            '"Yes", "No", or "No internet service"',
            '"Yes", "No", or "No internet service"',
            '"Yes", "No", or "No internet service"',
            '"Yes", "No", or "No internet service"',
            '"Yes", "No", or "No internet service"',
            '"Month-to-month", "One year", or "Two year"',
            '"Yes" or "No"',
            '"Electronic check", "Mailed check", "Bank transfer", or "Credit card"',
            'Number (e.g., 29.85)',
            'Number (e.g., 358.20)'
        ],
        'Will Be Converted To': [
            'Dropped',
            '0 or 1',
            '0 or 1',
            '0 or 1',
            '0 or 1',
            'As is',
            '0 or 1',
            '0 or 1',
            'One-hot encoded',
            '0 or 1',
            '0 or 1',
            '0 or 1',
            '0 or 1',
            '0 or 1',
            '0 or 1',
            'One-hot encoded',
            '0 or 1',
            'One-hot encoded',
            'As is',
            'As is (missing filled with median)'
        ]
    })
    
    st.dataframe(data_format)
    
    st.write("""
    ### Example Row:
    ```
    customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
    001,Female,0,Yes,No,12,Yes,No,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,358.20
    ```
    """)

# Other pages (Risk Dashboard, High Risk, Download Results) remain the same...
if selected == "Risk Dashboard":
    st.title("📈 Risk Dashboard")
    
    if st.session_state["predictions_df"] is not None:
        df = st.session_state["predictions_df"]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 HIGH", len(df[df.risk_level == "HIGH"]))
        col2.metric("🟠 MEDIUM", len(df[df.risk_level == "MEDIUM"]))
        col3.metric("🟡 LOW", len(df[df.risk_level == "LOW"]))
        col4.metric("🟢 VERY LOW", len(df[df.risk_level == "VERY LOW"]))
        
        fig, ax = plt.subplots()
        sns.countplot(x="risk_level", data=df,
                      order=["VERY LOW", "LOW", "MEDIUM", "HIGH"],
                      palette=["green", "yellow", "orange", "red"])
        plt.title("Risk Level Distribution")
        st.pyplot(fig)
    else:
        st.warning("Upload data first!")

if selected == "High Risk":
    st.title("🔥 High Risk Customers")
    if st.session_state["predictions_df"] is not None:
        df = st.session_state["predictions_df"]
        high = df[df.risk_level == "HIGH"].sort_values("churn_probability", ascending=False)
        st.dataframe(high.head(50))
    else:
        st.warning("Upload data first")

if selected == "Download Results":
    st.title("⬇ Download Predictions")
    if st.session_state["predictions_df"] is not None:
        df = st.session_state["predictions_df"]
        st.download_button("Download CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           "predictions.csv")
    else:
        st.info("No data available")