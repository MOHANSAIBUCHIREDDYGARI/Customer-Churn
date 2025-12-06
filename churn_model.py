import pandas as pd
import numpy as np
from datetime import datetime

class ChurnPredictionSystem:
    """Production-ready Churn Prediction System"""

    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.predictions_log = []

    def preprocess_customer_data(self, customer_df):
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in customer_df.columns:
                customer_df[feature] = 0

        # Select only the required features in correct order
        customer_df = customer_df[self.feature_names]
        return customer_df

    def predict_churn(self, customer_df):
        processed = self.preprocess_customer_data(customer_df)

        if self.scaler is not None:
            processed = self.scaler.transform(processed)

        predictions = self.model.predict(processed)
        probabilities = self.model.predict_proba(processed)[:, 1]

        return predictions, probabilities

    def batch_predict(self, customer_df):
        predictions, probabilities = self.predict_churn(customer_df)

        results = pd.DataFrame({
            "will_churn": predictions,
            "churn_probability": probabilities
        })

        def risk_level(p):
            if p >= 0.7: 
                return "HIGH"
            elif p >= 0.5: 
                return "MEDIUM"
            elif p >= 0.3: 
                return "LOW"
            else: 
                return "VERY LOW"

        results["risk_level"] = results["churn_probability"].apply(risk_level)
        
        return results