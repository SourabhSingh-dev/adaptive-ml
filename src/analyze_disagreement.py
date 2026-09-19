import pandas as pd
import numpy as np

def main():
    print("1. Loading model predictions...")
    df = pd.read_csv("../data/classical_model_predictions.csv")

    df["is_correct"] = (df['prediction'] == df['true_label']).astype(int)

    print("2. Building the Error Matrix...") 
    error_matrix = df.pivot(index='sample_id',columns='model_name',values='is_correct')

    print("\n--- Individual Model Accuracy ---")
    print((error_matrix.mean() * 100).round(2).astype(str) + "%")

    print("\n3. Calculating Oracle Potential...")
    error_matrix['oracle_correct'] = error_matrix.max(axis=1)
    oracle_accuracy = error_matrix['oracle_correct'].mean() * 100
    print(f"Oracle (Perfect Router) Accuracy: {oracle_accuracy:.2f}%")

    print("\n4. Analyzing Disagreement...")
    all_wrong = (error_matrix.sum(axis=1) == 0).sum()
    all_right = (error_matrix.sum(axis=1) == 3).sum()
    disagreement = len(error_matrix) - all_wrong - all_right
    print(f"Total Test Samples: {len(error_matrix)}")
    print(f"All Models Wrong: {all_wrong}")
    print(f"All Models Right: {all_right}")
    print(f"Disagreement (Router Opportunity): {disagreement} samples")

if __name__ == "__main__":
    main()