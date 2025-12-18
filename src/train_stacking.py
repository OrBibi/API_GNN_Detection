import os
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import project modules
from torch_geometric.loader import DataLoader
from graph_builder import build_graph_from_log
from vector_builder import build_vector_from_log 
from model import GNNClassifier
from features import FEATURE_DIM 

# --- Configuration ---
# Use ALL datasets
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
    """Loads the pre-trained GNN model."""
    print(f"Loading GNN from {path}...")
    model = GNNClassifier(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def get_calibrated_if_probs(if_package, X_vec):
    """Consistent IF probability calculation."""
    model = if_package['model']
    calib = if_package['calibration']
    global_min = calib['min']
    global_max = calib['max']
    
    scores = model.decision_function(X_vec)
    probs = (global_max - scores) / (global_max - global_min)
    probs = np.clip(probs, 0.0, 1.0)
    
    return probs

def generate_meta_features(df, gnn_model, rf_model, if_package, device):
    """Generates [GNN_Prob, RF_Prob, IF_Prob] for each log entry."""
    meta_features = []
    labels = []
    
    graph_batch = []
    vector_batch = []
    label_batch = []
    BATCH_SIZE = 512 
    
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        try:
            graph = build_graph_from_log(row['request'], row['response'], row['label'])
            gfv = build_vector_from_log(row['request'], row['response'])
            
            graph_batch.append(graph)
            vector_batch.append(gfv)
            label_batch.append(row['label'])
            
            if len(graph_batch) >= BATCH_SIZE:
                _process_batch(graph_batch, vector_batch, label_batch, 
                               gnn_model, rf_model, if_package, device, meta_features, labels)
                graph_batch, vector_batch, label_batch = [], [], []
                
        except Exception:
            continue
            
    if graph_batch:
        _process_batch(graph_batch, vector_batch, label_batch, 
                       gnn_model, rf_model, if_package, device, meta_features, labels)
        
    return np.array(meta_features), np.array(labels)

def _process_batch(graphs, vectors, labels, gnn_model, rf_model, if_package, device, meta_list, labels_list):
    # 1. GNN
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
    batch_data = next(iter(loader)).to(device)
    with torch.no_grad():
        out = gnn_model(batch_data.x, batch_data.edge_index, batch_data.batch)
        gnn_probs = torch.exp(out)[:, 1].cpu().numpy()

    # 2. RF
    X_vec = np.array(vectors)
    rf_probs = rf_model.predict_proba(X_vec)[:, 1]
    
    # 3. IF (Calibrated)
    if_probs = get_calibrated_if_probs(if_package, X_vec)

    # 4. Combine
    for i in range(len(labels)):
        meta_list.append([gnn_probs[i], rf_probs[i], if_probs[i]])
        labels_list.append(labels[i])

def train_stacking_ensemble():
    print(f"--- Starting Stacking Training (On ALL Validation Sets) ---")
    
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        gnn_model = load_gnn_model(GNN_MODEL_PATH, FEATURE_DIM, DEVICE)
        if_package = joblib.load(IF_PACKAGE_PATH)
        print("All base models loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    all_X_meta = []
    all_y = []

    # === STRATEGY: Train Stacking ONLY on Validation splits of 1, 2, 3, 4 ===
    splits_to_use = ['val'] 
    
    for ds in DATASETS:
        print(f" -> Processing {ds} [val]...")
        path = os.path.join(SPLITS_DIR, f"{ds}_val.parquet")
        
        if not os.path.exists(path): 
            print(f"    WARNING: {path} not found.")
            continue
        
        df = pd.read_parquet(path)
        X_meta_part, y_part = generate_meta_features(df, gnn_model, rf_model, if_package, DEVICE)
        all_X_meta.append(X_meta_part)
        all_y.append(y_part)

    if not all_X_meta:
        print("No data processed.")
        return

    X_meta_full = np.concatenate(all_X_meta, axis=0)
    y_full = np.concatenate(all_y, axis=0)

    print(f"\nFinal Meta-Dataset Shape: {X_meta_full.shape}")

    # Train Meta-Learner
    print("Training Logistic Regression...")
    meta_clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
    meta_clf.fit(X_meta_full, y_full)

    coeffs = meta_clf.coef_[0]
    print("\n--- Final Model Weights ---")
    print(f"GNN Weight: {coeffs[0]:.4f}")
    print(f"RF Weight:  {coeffs[1]:.4f}")
    print(f"IF Weight:  {coeffs[2]:.4f}")
    print(f"Intercept:  {meta_clf.intercept_[0]:.4f}")

    joblib.dump(meta_clf, STACKING_MODEL_PATH)
    print(f"\nSaved Stacking Model to {STACKING_MODEL_PATH}")

if __name__ == '__main__':
    train_stacking_ensemble()