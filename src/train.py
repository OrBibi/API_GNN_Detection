# src/train.py
import os
import torch
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple
from sklearn.metrics import recall_score, f1_score, accuracy_score # NEW: For advanced metrics

# Import project modules
from .graph_builder import build_graph_from_log
from .model import GNNClassifier 
from .features import FEATURE_DIM # Use the defined feature dimension (currently 64)

# --- Configuration ---
SPLITS_DIR = "data/processed/splits"
DATASET_NAME = 'dataset_3' # Keeping 'dataset_1' as requested
# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64

# --- Data Loading and Transformation ---

def load_and_transform_dataset(split: str, dataset_name: str) -> List:
    """
    Loads a Parquet split and transforms each log entry into a PyTorch Geometric Graph (Data object).
    """
    file_path = os.path.join(SPLITS_DIR, f"{dataset_name}_{split}.parquet")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Please check data/processed/splits.")
        return []

    print(f"Loading {dataset_name} ({split})...")
    df = pd.read_parquet(file_path)
    
    graph_list = []
    
    # tqdm provides a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Building {split} graphs"):
        try:
            request_data = row['request']
            response_data = row['response']
            label = row['label']
            
            # The core transformation using our builder
            graph = build_graph_from_log(request_data, response_data, label)
            graph_list.append(graph)
            
        except Exception as e:
            # Skipping row due to graph construction error
            continue
            
    print(f"Successfully transformed {len(graph_list)} graphs for {split}.")
    return graph_list

# --- Training and Evaluation Functions ---

def train(model, device, train_loader, optimizer):
    """Performs one training epoch."""
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        # NLLLoss is used because the model outputs log_softmax
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(model, device, loader) -> Tuple[float, float, float]:
    """
    Evaluates the model and returns Accuracy, Recall, and F1-Score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Get the predicted class
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    # Recall for the positive class (class 1: Attack). Measures true attack detection rate.
    rec = recall_score(all_labels, all_preds, zero_division=0, pos_label=1) 
    # F1-Score is the harmonic mean of precision and recall.
    f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

    return acc, rec, f1 # Returns Accuracy, Recall, F1

# --- Main Execution ---

if __name__ == '__main__':
    print(f"Starting GNN Training on {DATASET_NAME} using device: {DEVICE}")

    # 1. Load Data
    train_graphs = load_and_transform_dataset('train', DATASET_NAME)
    val_graphs = load_and_transform_dataset('val', DATASET_NAME)

    if not train_graphs or not val_graphs:
        print("Cannot continue: Data loading failed or resulted in empty lists.")
    else:
        # 2. Create DataLoaders
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

        # 3. Initialize Model and Optimizer
        # FEATURE_DIM is now 64 based on the latest src/features.py update
        model = GNNClassifier(feature_dim=FEATURE_DIM).to(DEVICE) 
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 4. Training Loop
        best_val_f1 = 0 # Track F1 score for saving the best model
        for epoch in range(1, EPOCHS + 1):
            loss = train(model, DEVICE, train_loader, optimizer)
            
            # Evaluate on both sets
            train_acc, train_rec, train_f1 = evaluate(model, DEVICE, train_loader)
            val_acc, val_rec, val_f1 = evaluate(model, DEVICE, val_loader)
            
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Train Rec: {train_rec:.4f}, Train F1: {train_f1:.4f} | '
                  f'Val Acc: {val_acc:.4f}, Val Rec: {val_rec:.4f}, Val F1: {val_f1:.4f}')

            # Save the best model based on validation F1-score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f'models/{DATASET_NAME}_best_model.pth')
                print("--> Saved best model state.")

        print("\nTraining completed.")
        print(f"Best Validation F1-Score achieved: {best_val_f1:.4f}")