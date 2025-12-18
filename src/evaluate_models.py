import os
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch_geometric.loader import DataLoader

# Import project modules
from graph_builder import build_graph_from_log
from vector_builder import build_vector_from_log 
from model import GNNClassifier
from features import FEATURE_DIM 

# --- Configuration ---
# Evaluate on ALL Test Sets (1, 2, 3, 4)
DATASETS = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']
SPLITS_DIR = "data/processed/splits"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_OUTPUT_DIR = 'models'

# Model Paths
RF_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'combined_rf_model.pkl')
IF_PACKAGE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'combined_if_model_calibrated.pkl')
GNN_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'final_combined_gnn_model.pth')
STACKING_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'final_stacking_model.pkl')

def load_gnn_model(path, feature_dim, device):
    """Loads the pre-trained GNN model from disk."""
    print(f"Loading GNN from {path}...")
    model = GNNClassifier(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def prepare_data_for_evaluation(df, dataset_name):
    """Transforms raw log data into Graphs and Vectors."""
    graph_list = []
    tabular_features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}", leave=False):
        try:
            graph = build_graph_from_log(row['request'], row['response'], row['label'])
            gfv = build_vector_from_log(row['request'], row['response'])
            
            graph_list.append(graph)
            tabular_features.append(gfv)
            labels.append(row['label'])
        except Exception:
            continue
            
    return graph_list, np.array(tabular_features), np.array(labels)

def get_gnn_probabilities(model, graphs, device):
    if not graphs: return np.array([])
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for data in tqdm(loader, desc="GNN Inference"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.exp(out)[:, 1] 
            all_probs.extend(probs.cpu().tolist())
    return np.array(all_probs)

def get_calibrated_if_probs(if_package, X_vec):
    model = if_package['model']
    calib = if_package['calibration']
    global_min = calib['min']
    global_max = calib['max']
    scores = model.decision_function(X_vec)
    probs = (global_max - scores) / (global_max - global_min)
    return np.clip(probs, 0.0, 1.0)

def print_full_report(y_true, y_probs, model_name):
    """Generates a detailed performance report for a specific model."""
    y_pred = (y_probs > 0.5).astype(int)
    
    print(f"\n{'='*20} {model_name.upper()} RESULTS {'='*20}")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack'], digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # KPIs
    tn, fp, fn, tp = cm.ravel()
    attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    
    print("-" * 50)
    print(f"Global Accuracy:                {accuracy:.2%}")
    print(f"Attack Detection (Recall):      {attack_recall:.2%}")
    print(f"False Alarm Rate (FPR):         {fpr:.2%}")
    print(f"Precision on Attacks:           {attack_precision:.2%}")
    print("-" * 50)

def evaluate_final_test():
    print(f"--- Starting Comparative Evaluation (On ALL Test Sets) ---")
    
    # 1. Load All Models
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        gnn_model = load_gnn_model(GNN_MODEL_PATH, FEATURE_DIM, DEVICE)
        if_package = joblib.load(IF_PACKAGE_PATH)
        stacking_model = joblib.load(STACKING_MODEL_PATH)
        print(" [v] All models loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # 2. Load and Prepare Data
    all_graphs = []
    all_X_tab = []
    all_labels = []

    # Strategy: Load 'test' splits from 1, 2, 3, 4
    splits_to_use = ['test']

    for ds in DATASETS:
        for split in splits_to_use:
            file_path = os.path.join(SPLITS_DIR, f"{ds}_{split}.parquet")
            
            if not os.path.exists(file_path):
                print(f"Skipping {ds}_{split} (File not found).")
                continue

            print(f"Loading {ds} [{split}]...")
            df_raw = pd.read_parquet(file_path)
            
            graphs, X_tab, y_true = prepare_data_for_evaluation(df_raw, f"{ds}_{split}")
            
            if len(y_true) > 0:
                all_graphs.extend(graphs)
                all_X_tab.append(X_tab)
                all_labels.append(y_true)

    if not all_labels:
        print("No valid data found for evaluation.")
        return

    # Combine data
    X_full = np.concatenate(all_X_tab, axis=0)
    y_full = np.concatenate(all_labels, axis=0)
    print(f"\nTotal Samples for Evaluation: {len(y_full)}")

    # 3. Inference & Reporting per Model
    
    # --- A. GNN Only ---
    gnn_probs = get_gnn_probabilities(gnn_model, all_graphs, DEVICE)
    print_full_report(y_full, gnn_probs, "GNN Only")

    # --- B. Random Forest Only ---
    print("\nRunning Random Forest Inference...")
    rf_probs = rf_model.predict_proba(X_full)[:, 1]
    print_full_report(y_full, rf_probs, "Random Forest")

    # --- C. Isolation Forest Only ---
    print("\nRunning Isolation Forest Inference...")
    if_probs = get_calibrated_if_probs(if_package, X_full)
    print_full_report(y_full, if_probs, "Isolation Forest")
    
    # --- D. Stacking Ensemble ---
    print("\nRunning Stacking Ensemble Inference...")
    X_meta = np.column_stack((gnn_probs, rf_probs, if_probs))
    final_probs = stacking_model.predict_proba(X_meta)[:, 1]
    
    print_full_report(y_full, final_probs, "Stacking Ensemble (Final)")

if __name__ == '__main__':
    evaluate_final_test()