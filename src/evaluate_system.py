import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from data.preprocessing import extract_tabular_features

def main():
    print("1. Loading Test Data and Pre-calculated Predictions...")
    data = np.load("../data/processed/hhar_windows.npz", allow_pickle=True)
    X_seq = data['X']
    y = data['y']
    meta = data['meta_list']
    
    classical_df = pd.read_csv("../data/classical_model_predictions.csv")
    deep_df = pd.read_csv("../data/deep_model_predictions.csv")
    df = pd.concat([classical_df, deep_df], ignore_index=True)
    
    df['is_correct'] = (df['prediction'] == df['true_label']).astype(int)
    error_matrix = df.pivot(index='sample_id', columns="model_name", values='is_correct')

    print("2. Reconstructing Unseen Test Set...")
    user_groups = meta[:,0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X_seq, y, groups=user_groups))

    X_test_seq = X_seq[test_idx]
    X_test_features = extract_tabular_features(X_test_seq)
    test_error_matrix = error_matrix.reset_index(drop=True)

    print("3. Loading the Trained Router...")
    router = joblib.load("../models/router_xgb.pkl")
    le = joblib.load("../models/router_label_encoder.pkl")
    inverse_model_map = {i: name for i, name in enumerate(le.classes_)}

    print("4. Simulating Production Inference...")

    router_preds = router.predict(X_test_features)
    routed_model_names = [inverse_model_map[p] for p in router_preds]

    cost_map = {
        'Logistic_Regression': 1.0,   
        'XGBoost': 2.41,
        '1D_CNN': 3.69,               
        'LSTM': 3.37
    }


    acc_lr = test_error_matrix['Logistic_Regression'].mean()
    cost_lr = cost_map['Logistic_Regression']


    acc_lstm = test_error_matrix['LSTM'].mean()
    cost_lstm = cost_map['LSTM']


    routed_correct = 0
    routed_cost = 0
    for i, model_name in enumerate(routed_model_names):
        routed_correct += test_error_matrix.iloc[i][model_name]
        routed_cost += cost_map[model_name]

    acc_router = routed_correct / len(router_preds)
    avg_cost_router = routed_cost / len(router_preds)

    print("\n" + "=" * 55)
    print(" FINAL SYSTEM PERFORMANCE COMPARISON ")
    print("=" * 55)
    print(f"1. Static Monolith (Always Logistic):")
    print(f"   Accuracy = {acc_lr*100:.1f}% | Avg Latency Cost = {cost_lr:.2f}\n")
    print(f"2. Static Monolith (Always LSTM):")
    print(f"   Accuracy = {acc_lstm*100:.1f}% | Avg Latency Cost = {cost_lstm:.2f}\n")
    print(f"3. AdaptiveML (Our Smart Router):")
    print(f"   Accuracy = {acc_router*100:.1f}% | Avg Latency Cost = {avg_cost_router:.2f}")
    print("=" * 55)

if __name__ == "__main__":
    main()