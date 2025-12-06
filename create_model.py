# create_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

print("="*70)
print("CREATING COMPATIBLE CHURN PREDICTION MODEL")
print("="*70)

# Load your dataset
print("Loading dataset...")
df = pd.read_csv("C:/Users/mohansai/OneDrive/Desktop/PROJECTS/Customer Churn/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Simple preprocessing (similar to your notebook)
df_clean = df.copy()
df_clean = df_clean.drop(['customerID'], axis=1)
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)

# Binary encoding
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

# Service columns
service_map = {'No': 0, 'Yes': 1, 'No phone service': 0, 'No internet service': 0}
service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].map(service_map)

# One-hot encoding
cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

# Prepare features
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Import and create ChurnPredictionSystem
try:
    from churn_model import ChurnPredictionSystem
    print("✓ Imported ChurnPredictionSystem")
    
    # Create the system
    churn_system = ChurnPredictionSystem(
        model=model,
        scaler=scaler,
        feature_names=X.columns.tolist()
    )
    
    # Save it
    with open("churn_system.pkl", "wb") as f:
        pickle.dump(churn_system, f)
    
    print("✓ Model saved as 'churn_system.pkl'")
    
    # Test it
    test_score = model.score(X_test, y_test)
    print(f"✓ Test accuracy: {test_score:.4f}")
    print(f"✓ Features: {len(X.columns)}")
    print(f"✓ File saved successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    # Create ChurnPredictionSystem class inline if import fails
    class ChurnPredictionSystem:
        def __init__(self, model, scaler, feature_names):
            self.model = model
            self.scaler = scaler
            self.feature_names = feature_names
        
        def preprocess_customer_data(self, customer_df):
            for feature in self.feature_names:
                if feature not in customer_df.columns:
                    customer_df[feature] = 0
            customer_df = customer_df[self.feature_names]
            return customer_df
        
        def batch_predict(self, customer_df):
            processed = self.preprocess_customer_data(customer_df)
            if self.scaler is not None:
                processed = self.scaler.transform(processed)
            
            predictions = self.model.predict(processed)
            probabilities = self.model.predict_proba(processed)[:, 1]
            
            results = pd.DataFrame({
                "will_churn": predictions,
                "churn_probability": probabilities
            })
            
            def risk_level(p):
                if p >= 0.7: return "HIGH"
                elif p >= 0.5: return "MEDIUM"
                elif p >= 0.3: return "LOW"
                else: return "VERY LOW"
            
            results["risk_level"] = results["churn_probability"].apply(risk_level)
            return results
    
    churn_system = ChurnPredictionSystem(model, scaler, X.columns.tolist())
    
    with open("churn_system.pkl", "wb") as f:
        pickle.dump(churn_system, f)
    
    print("✓ Model saved with inline class definition")

print("\n" + "="*70)
print("✅ MODEL CREATION COMPLETE!")
print("Now run: streamlit run app.py")
print("="*70)