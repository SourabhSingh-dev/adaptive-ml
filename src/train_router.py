import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from xgboost import XGBClassifier
from data.preprocessing import extract_tabular_features
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

def main():
    print("1. Loading raw data and predictions...")
    data = np.load("../data/processed/hhar_windows.npz", allow_pickle=True)
    X_seq = data['X']
    y = data['y']
    meta = data['meta_list']
    
    classical_df = pd.read_csv("../data/classical_model_predictions.csv")
    deep_df = pd.read_csv("../data/deep_model_predictions.csv")
    df = pd.concat([classical_df, deep_df], ignore_index=True)

    print("2. Reconstructing the test features...")
    user_groups = meta[:,0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X_seq,y,groups=user_groups))

    X_test_seq = X_seq[test_idx]
    X_test_features = extract_tabular_features(X_test_seq)

    print("3. Building the target (Best Model) based on Cost Priority...")
    df['is_correct'] = (df['prediction'] == df['true_label']).astype(int)
    error_matrix = df.pivot(index='sample_id',columns="model_name",values='is_correct')

    cost_map = {
        'Logistic_Regression': 1.0,   
        'Random_Forest': 123.84,   
        'XGBoost': 2.41,    
        '1D_CNN': 3.69,               
        'LSTM': 3.37
    }
    COST_PENALTY = 0.01

    best_models = []
    for index,row in error_matrix.iterrows():
        chosen = 'None'
        best_utility = -float('inf')

        available_models = [m for m in cost_map.keys() if m in row.index]
        if row[available_models].sum() > 0:
            for model in available_models:
                utility = row[model] - (COST_PENALTY * cost_map[model])

                if utility > best_utility:
                    best_utility = utility
                    chosen = model
        best_models.append(chosen)

    error_matrix['target_model'] = best_models

    valid_mask = error_matrix['target_model'] != 'None'
    valid_samples = error_matrix[valid_mask].copy()
    valid_features = X_test_features[valid_mask.values]

    le = LabelEncoder()
    y_router = le.fit_transform(valid_samples['target_model'])
    inverse_model_map = {i: name for i, name in enumerate(le.classes_)}

    print("4. Training the Meta-Learner (Router)...")
    valid_users = user_groups[test_idx][valid_mask.values]

    router_gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    router_train_idx, router_test_idx = next(router_gss.split(valid_features, y_router, groups=valid_users))

    X_rt_train = valid_features.iloc[router_train_idx]
    X_rt_test = valid_features.iloc[router_test_idx]
    y_rt_train = y_router[router_train_idx]
    y_rt_test = y_router[router_test_idx] 

    unique_train, counts_train = np.unique(y_rt_train, return_counts=True)
    print("\nActual Target Distribution in Training Data:")
    for u, c in zip(unique_train, counts_train):
        print(f"{inverse_model_map[u]}: {c} times ({(c/len(y_rt_train))*100:.1f}%)")

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_rt_train)
    router = XGBClassifier(
        n_estimators = 100, max_depth = 4, learning_rate = 0.1, random_state = 42
    )
    router.fit(X_rt_train, y_rt_train,sample_weight=sample_weights)

    print("5. Evaluating Router Performance...")
    router_preds = router.predict(X_rt_test)

    acc = (router_preds == y_rt_test).mean()
    print(f"Router Meta-Accuracy (Choosing the correct model): {acc * 100:.2f}%")

    unique, counts = np.unique(router_preds, return_counts=True)
    print("\nRouter Selection Distribution:")

    for u,c in zip(unique,counts):
        print(f"{inverse_model_map[u]}: {c} times ({(c/len(router_preds))*100:.1f}%)")

    print("\n6. Saving the Meta-Learner (Router) artifacts...")
    os.makedirs("../models", exist_ok=True)
    joblib.dump(router, "../models/router_xgb.pkl")
    joblib.dump(le, "../models/router_label_encoder.pkl")
    print("Router saved successfully to the models folder!")
if __name__  == "__main__":
    main()