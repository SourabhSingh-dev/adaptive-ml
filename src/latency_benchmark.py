import time
import torch
import joblib
import numpy as np
from deep_models import SensorCNN, SensorLSTM

def benchmark_model(model, dummy_input, is_pytorch=False, device="cpu"):

    for _ in range(10):
        if is_pytorch:
            with torch.no_grad():
                _ = model(dummy_input)
        else:
            _ = model.predict(dummy_input)


    start_time = time.perf_counter()
    runs = 100
    for _ in range(runs):
        if is_pytorch:
            with torch.no_grad():
                _ = model(dummy_input)
        else:
            _ = model.predict(dummy_input)
    end_time = time.perf_counter()
    

    return ((end_time - start_time) / runs) * 1000

def main():
    print("1. Loading saved models...")

    lr_model = joblib.load("../models/logistic_regression.joblib")
    rf_model = joblib.load("../models/random_forest.joblib")
    xgb_model = joblib.load("../models/XGBoost.joblib")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = SensorCNN(in_channels=3, num_classes=6).to(device)
    cnn_model.load_state_dict(torch.load("../models/1d_cnn.pt", weights_only=True))
    cnn_model.eval()

    lstm_model = SensorLSTM(input_size=3, hidden_size=64, num_layers=1, num_classes=6).to(device)
    lstm_model.load_state_dict(torch.load("../models/lstm.pt", weights_only=True))
    lstm_model.eval()

    print("2. Generating single-sample dummy data...")
    
    num_tabular_features = lr_model.n_features_in_ if hasattr(lr_model, 'n_features_in_') else 12
    dummy_tabular = np.random.rand(1, num_tabular_features)

    dummy_tensor_cnn = torch.randn(1, 128, 3).to(device)
    dummy_tensor_lstm = torch.randn(1, 128, 3).to(device)

    print("\n3. Running latency benchmarks (milliseconds per sample)...")
    latencies = {
        "Logistic_Regression": benchmark_model(lr_model, dummy_tabular, is_pytorch=False),
        "Random_Forest": benchmark_model(rf_model, dummy_tabular, is_pytorch=False),
        "XGBoost": benchmark_model(xgb_model, dummy_tabular, is_pytorch=False),
        "1D_CNN": benchmark_model(cnn_model, dummy_tensor_cnn, is_pytorch=True, device=device),
        "LSTM": benchmark_model(lstm_model, dummy_tensor_lstm, is_pytorch=True, device=device)
    }

    for name, lat in latencies.items():
        print(f"-> {name}: {lat:.4f} ms")

    print("\n4. Calculating Router Cost Multipliers (Normalized to LR)...")
    base_time = latencies["Logistic_Regression"]
    cost_map = {name: round(lat / base_time, 2) for name, lat in latencies.items()}
    
    for name, cost in cost_map.items():
        print(f"'{name}': {cost},")

if __name__ == "__main__":
    main()