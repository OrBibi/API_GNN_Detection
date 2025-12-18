# src/train.py
import os
import torch
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data # ADDED: Used for graph objects
from typing import List, Dict, Tuple
from sklearn.metrics import recall_score, f1_score, accuracy_score 

# Import project modules
from graph_builder import build_graph_from_log
from model import GNNClassifier
from features import FEATURE_DIM 
# Assuming data_loader is available for SPLITS_DIR
from data_loader import SPLITS_DIR # ADDED: For modularity

# --- Configuration (UPDATED for Combined Training) ---
DATASETS_TO_COMBINE = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4'] # NEW: List of all datasets
FINAL_MODEL_NAME = 'final_combined_gnn_model.pth' # NEW: Fixed name for the final saved model

# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
MODEL_OUTPUT_DIR = 'models'

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# --- Data Loading and Transformation (UPDATED for Combined Loading) ---

def load_and_transform_dataset(split: str, datasets: List[str]) -> List[Data]:
    """
    Loads Parquet splits for multiple datasets, transforms all logs into graphs, 
    and returns a single combined list of PyTorch Geometric Data objects.
    """
    all_graphs: List[Data] = [] # Initialize the combined list
    
    for dataset_name in datasets: # Loop through each dataset name
        
        file_path = os.path.join(SPLITS_DIR, f"{dataset_name}_{split}.parquet")
        
        if not os.path.exists(file_path):
            print(f"Error: File not found for {dataset_name} at {file_path}. Skipping.")
            continue
        
        print(f"Loading {dataset_name} ({split})...")
        df = pd.read_parquet(file_path)
        
        graphs_in_current_set = 0
        
        # tqdm provides a progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Building {dataset_name} {split} graphs"): 
            try:
                request_data = row['request']
                response_data = row['response']
                label = row['label']
                
                # CORRECT CALL: build_graph_from_log returns a single graph object
                graph = build_graph_from_log(request_data, response_data, label)
                
                all_graphs.append(graph)
                graphs_in_current_set += 1
                
            except Exception as e:
                # print(f"CRITICAL ERROR in log parsing: {e}") # Uncomment for deeper debug
                continue 
                
        print(f"INFO: Added {graphs_in_current_set} graphs from {dataset_name}.")
        # DEBUG PRINT: Check the total size of the list after each dataset is processed
        print(f"DEBUG: Total graphs in list after {dataset_name}: {len(all_graphs)}") 
        
    return all_graphs # Return the single combined list

# --- Training and Evaluation Functions (No change) ---

def train(model, device, train_loader, optimizer):
    """Performs one training epoch."""
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
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
            pred = out.argmax(dim=1) 
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0, pos_label=1) 
    f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

    return acc, rec, f1 

# --- Main Execution (UPDATED for Combined Training) ---

if __name__ == '__main__':
    print(f"Starting Combined GNN Training on datasets: {DATASETS_TO_COMBINE} using device: {DEVICE}")

    # 1. Load Data
    train_graphs = load_and_transform_dataset('train', DATASETS_TO_COMBINE) 
    val_graphs = load_and_transform_dataset('val', DATASETS_TO_COMBINE) 

    if not train_graphs or not val_graphs:
        print(f"DEBUG FINAL CHECK: train_graphs size: {len(train_graphs)}, val_graphs size: {len(val_graphs)}")
        print("Cannot continue: Data loading failed or resulted in empty lists.")
    else:
        print(f"Total training samples for GNN: {len(train_graphs)}")
        print(f"Total validation samples for GNN: {len(val_graphs)}")
        
        # 2. Create DataLoaders
        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

        # 3. Initialize Model and Optimizer
        model = GNNClassifier(feature_dim=FEATURE_DIM).to(DEVICE) 
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 4. Training Loop
        best_val_f1 = 0 
        
        print(f"INFO: Starting training with target save name: {FINAL_MODEL_NAME}")
        
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
                torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, FINAL_MODEL_NAME)) 
                print(f"--> Saved best model state to {FINAL_MODEL_NAME}.")

        print("\nTraining completed.")
        print(f"Best Validation F1-Score achieved: {best_val_f1:.4f}")