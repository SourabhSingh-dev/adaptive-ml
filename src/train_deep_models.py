import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from deep_models import SensorCNN, SensorLSTM
torch.manual_seed(42)
np.random.seed(42)

def main():

    print("1. Loading pre-processed 3D tensors...")
    data = np.load("../data/processed/hhar_windows.npz", allow_pickle=True)
    X = data['X']
    y = data['y']
    meta = data['meta_list']

    print("2. Recreating the exact Group-Aware Split...")
    user_groups = meta[:,0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
    train_idx, test_idx = next(gss.split(X,y,groups=user_groups))

    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

    print("3. Building PyTorch DataLoaders...")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t,y_train_t),batch_size=64,shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t,y_test_t),batch_size=64,shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using hardware accelerator: {device}")

    print("\n4. Initializing Deep Learning Models...")
    models = {
        "1D_CNN" : SensorCNN(in_channels=3,num_classes=6).to(device),
        "LSTM" : SensorLSTM(input_size=3,hidden_size=64,num_layers=1,num_classes=6).to(device)
    }

    results_list = []

    for name,model in models.items():
        print(f"\n--- Training {name} ---")
        criterion = nn.CrossEntropyLoss()
        optmizer = optim.Adam(model.parameters(),lr=0.001)

        epochs = 5
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x,batch_y = batch_x.to(device), batch_y.to(device)

                optmizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs,batch_y)
                loss.backward()
                optmizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

        print(f"Evaluating {name} on unseen test users...")
        model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)

                probs = torch.softmax(outputs,dim=1)
                preds = torch.argmax(probs,dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        acc = (np.array(all_preds) == y_test).mean()
        print(f"{name} Final Accuracy: {acc * 100:.2f}%")

        for i in range(len(y_test)):
            results_list.append({
                "sample_id" : i,
                "true_label" : y_test[i],
                "model_name" : name,
                "prediction" : all_preds[i],
                "confidence" : np.max(all_probs[i])
            })

    print("\n5. Saving deep learning predictions for the Router...")
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("../data/deep_model_predictions.csv", index=False)
    print("Done! Saved to data/deep_model_predictions.csv")

    print("\n6. Saving trained model artifacts...")
    
    os.makedirs("../models", exist_ok=True)
    

    torch.save(models['LSTM'].state_dict(), "../models/lstm.pt")
    torch.save(models["1D_CNN"].state_dict(), "../models/1d_cnn.pt")
    
    
    print("Models saved successfully!")
if __name__ == "__main__":
    main()