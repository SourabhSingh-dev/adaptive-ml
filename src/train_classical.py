import polars as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from data.preprocessing import create_sliding_windows,extract_tabular_features

def main():
    print("1. Loading raw dataset...")
    df = pl.read_csv("../data/raw/Phones_accelerometer.csv")

    print("2. Generating 2.5-second windows...")
    X_seq,y,meta = create_sliding_windows(df)

    print("3. Extracting tabular features...")
    X_tab = extract_tabular_features(X_seq)

    print("4. Performing Group-Aware Split (Isolating Users)...")
    user_groups = meta[:,0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx,test_idx = next(gss.split(X_tab,y,groups=user_groups))


    X_train,X_test = X_tab.iloc[train_idx],X_tab.iloc[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

    print("-> Applying Feature Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    unique_classes = np.unique(y)
    label_map = {label : idx for idx,label in enumerate(unique_classes)}
    y_train_num = np.array([label_map[label] for label in y_train])
    y_test_num = np.array([label_map[label] for label in y_test])

    print("\n5. Training Candidate Models...")
    models = {
        "Logistic_Regression" : LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random_Forest" : RandomForestClassifier(
            n_estimators=100, max_depth=10,min_samples_leaf=5 ,random_state=42, n_jobs=-1
        ),
        "XGBoost" : XGBClassifier(
            n_estimators=100,max_depth=4,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1
        )
    }

    results_list = []

    for name,model in models.items():
        print(f"-> Training {name}...")
        model.fit(X_train_scaled, y_train_num)

        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)

        acc = (predictions == y_test_num).mean()
        print(f"Accuracy: {acc * 100:.2f}%")

        for i in range(len(y_test_num)):
            results_list.append({
                "sample_id" : i,
                "true_label" : y_test_num[i],
                "model_name" : name,
                "prediction" : predictions[i],
                "confidence" : np.max(probabilities[i])
            })
    print("\n6. Saving prediction results for the Router...")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("../data/classical_model_predictions.csv", index=False)
    print("Done! Saved to data/classical_model_predictions.csv")

if __name__ == "__main__":
    main()