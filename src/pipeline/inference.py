import torch
import numpy as np
import joblib
import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.preprocessing import extract_tabular_features 
from deep_models import SensorCNN, SensorLSTM

class AdaptiveInferencePipeline:
    def __init__(self, models_dir="../../models"):

        print("Loading AdaptiveML System into memory...")
        self.router = joblib.load(f"{models_dir}/router_xgb.pkl")
        self.label_encoder = joblib.load(f"{models_dir}/router_label_encoder.pkl")

        cnn_model = SensorCNN(in_channels=3,num_classes=6) 
        lstm_model = SensorLSTM(input_size=3,hidden_size=64,num_layers=1,num_classes=6)

        cnn_model.load_state_dict(torch.load(f"{models_dir}/1d_cnn.pt", weights_only=True))
        lstm_model.load_state_dict(torch.load(f"{models_dir}/lstm.pt", weights_only=True))
        
        cnn_model.eval()
        lstm_model.eval()

        self.models = {
            'Logistic_Regression': joblib.load(f"{models_dir}/logistic_regression.joblib"),
            'XGBoost': joblib.load(f"{models_dir}/XGBoost.joblib"),
            'Random_Forest': joblib.load(f"{models_dir}/random_forest.joblib"),
            '1D_CNN': cnn_model,
            'LSTM': lstm_model
        }
        
        self.inverse_model_map = {i: name for i, name in enumerate(self.label_encoder.classes_)}
        self.activity_map = {
            0: 'bike',
            1: 'sit',
            2: 'stairsdown',
            3: 'stairsup',
            4: 'stand',
            5: 'walk'
        }

    def predict(self, raw_sensor_window):

        tabular_features = extract_tabular_features(raw_sensor_window)
        
        router_pred_idx = self.router.predict(tabular_features)[0]
        selected_model_name = self.inverse_model_map[router_pred_idx]
        selected_model = self.models[selected_model_name]
        
        if selected_model_name in ['Logistic_Regression', 'XGBoost', 'Random_Forest']:
            activity_prediction = selected_model.predict(tabular_features)[0]
        else:
            tensor_data = torch.tensor(raw_sensor_window, dtype=torch.float32)
            
            with torch.no_grad():
                logits = selected_model(tensor_data)
                activity_prediction = torch.argmax(logits, dim=1).item()
            
        return {
            "routed_to": selected_model_name,
            "predicted_activity": self.activity_map[int(activity_prediction)]
        }

if __name__ == "__main__":
    pipeline = AdaptiveInferencePipeline()
    
    dummy_data = np.random.randn(1, 128, 3)
    
    result = pipeline.predict(dummy_data)
    print(f"Incoming Sensor Data Routed To: {result['routed_to']}")
    print(f"Final Activity Prediction: {result['predicted_activity']}")