import os
import torch
import joblib
import pandas as pd
import numpy as np
import sys
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch_geometric.loader import DataLoader

# Ensure project modules are findable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_builder import build_graph_from_log
from vector_builder import build_vector_from_log 
from model import GNNClassifier
from features import FEATURE_DIM 

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- Configuration ---
DATASETS = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']
SPLITS_DIR = "data/processed/splits"
DEVICE = torch.device('cpu') 
MODEL_OUTPUT_DIR = 'models'
BATCH_SIZE = 1024 # For GNN Inference

# Model Paths
RF_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'combined_rf_model.pkl')
IF_PACKAGE_PATH = os.path.join(MODEL_OUTPUT_DIR, 'combined_if_model_calibrated.pkl')
GNN_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'final_combined_gnn_model.pth')
STACKING_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'final_stacking_model.pkl')

def fast_evaluate():
    print(f"--- Starting Fast Deterministic Evaluation (Full Report) ---")
    
    # 1. Load Models
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        if_package = joblib.load(IF_PACKAGE_PATH)
        stacking_model = joblib.load(STACKING_MODEL_PATH)
        
        gnn_model = GNNClassifier(feature_dim=FEATURE_DIM).to(DEVICE)
        gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=DEVICE))
        gnn_model.eval()
        print(" [v] All models loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODELS: {e}")
        return

    all_gnn_probs = []
    all_rf_probs = []
    all_if_probs = []
    all_labels = []

    # 2. Process Datasets
    for ds in DATASETS:
        file_path = os.path.join(SPLITS_DIR, f"{ds}_test.parquet")
        if not os.path.exists(file_path): 
            print(f"Skipping {ds} - file not found.")
            continue

        print(f"\nProcessing {ds}...")
        df = pd.read_parquet(file_path)
        
        graphs = []
        vectors = []
        
        # Building Graphs and Vectors
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building {ds}"):
            graphs.append(build_graph_from_log(row['request'], row['response'], row['label']))
            vectors.append(build_vector_from_log(row['request'], row['response']))
            all_labels.append(row['label'])

        # GNN Inference 
        loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"GNN Inference {ds}"):
                batch = batch.to(DEVICE)
                out = gnn_model(batch.x, batch.edge_index, batch.batch)
                probs = torch.exp(out)[:, 1].cpu().tolist()
                all_gnn_probs.extend(probs)

        # RF & IF Inference
        X_tab = np.array(vectors)
        # Random Forest Probabilities
        all_rf_probs.extend(rf_model.predict_proba(X_tab)[:, 1])
        
        # Isolation Forest Probabilities (Calibrated)
        if_model = if_package['model']
        calib = if_package['calibration']
        if_scores = if_model.decision_function(X_tab)
        if_p = np.clip((calib['max'] - if_scores) / (calib['max'] - calib['min']), 0.0, 1.0)
        all_if_probs.extend(if_p)

    # 3. Stacking Ensemble Inference
    print("\nRunning Stacking Ensemble Inference...")
    X_meta = np.column_stack((all_gnn_probs, all_rf_probs, all_if_probs))
    all_final_probs = stacking_model.predict_proba(X_meta)[:, 1]

    # 4. Detailed Reports
    y_true = np.array(all_labels)
    
    def print_report(probs, name):
        preds = (np.array(probs) > 0.5).astype(int)
        print("\n" + "="*20 + f" {name.upper()} " + "="*20)
        print(classification_report(y_true, preds, target_names=['Benign', 'Attack'], digits=4))
        print(f"Confusion Matrix:\n{confusion_matrix(y_true, preds)}")

    # Printing report for each component
    print_report(all_gnn_probs, "GNN Only")
    print_report(all_rf_probs, "Random Forest Only")
    print_report(all_if_probs, "Isolation Forest Only")
    print_report(all_final_probs, "Final Stacking Ensemble")

if __name__ == '__main__':
    fast_evaluate()