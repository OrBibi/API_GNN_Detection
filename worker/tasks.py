import os
import torch
import joblib
import numpy as np
import sys
import json

# --- 1. System Path Fix (CRITICAL) ---
# This ensures that when we import from src, those modules can find 'features.py'
# which sits directly inside /app/src/
sys.path.append('/app')
sys.path.append('/app/src')

# --- 2. Import EXISTING Builders ---
# We use the exact code that was used for training
from src.graph_builder import build_graph_from_log
from src.vector_builder import build_vector_from_log 
from src.model import GNNClassifier
from src.features import FEATURE_DIM 

# --- 3. Configuration ---
MODEL_DIR = "/app/models"
DEVICE = torch.device('cpu')

# --- 4. Load Models (Once) ---
RF_MODEL = joblib.load(os.path.join(MODEL_DIR, 'combined_rf_model.pkl'))
IF_PACKAGE = joblib.load(os.path.join(MODEL_DIR, 'combined_if_model_calibrated.pkl'))

# Load GNN structure and weights
GNN_MODEL = GNNClassifier(feature_dim=FEATURE_DIM).to(DEVICE)
GNN_MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'final_combined_gnn_model.pth'), map_location=DEVICE))
GNN_MODEL.eval()

STACKING_MODEL = joblib.load(os.path.join(MODEL_DIR, 'final_stacking_model.pkl'))

def process_request(full_log):
    """
    Processing task using the ORIGINAL source code builders.
    Maps input data to strictly match the Training Dataset schema.
    """
    try:
        # Extract raw data from the input (which comes from index.html/main.py)
        raw_req = full_log.get('request', {})
        raw_res = full_log.get('response', {})

        # --- DATA MAPPING STRATEGY ---
        # Based on actual training data from parquet:
        # Request: method, url, headers, body (string)
        # Response: status_code, status, headers, body (string)
        
        request_obj = {
            "method": str(raw_req.get('method', 'GET')),
            "url": str(raw_req.get('url', '')),
            "headers": raw_req.get('headers', {}),
            "body": str(raw_req.get('body', ''))
        }

        response_obj = {
            "status_code": int(raw_res.get('status_code', 200)),
            "status": str(raw_res.get('status', '')),
            "headers": raw_res.get('headers', {}),
            "body": str(raw_res.get('body', ''))
        }


        # --- 5. USE EXISTING BUILDERS ---
        # No local logic. We trust the src code.
        
        # Build Graph (for GNN)
        graph = build_graph_from_log(request_obj, response_obj, label=0)
        
        # Build Vector (for RF/IF)
        gfv = build_vector_from_log(request_obj, response_obj)
        X_tab = np.array([gfv])

        # --- 6. INFERENCE ---
        
        # GNN
        with torch.no_grad():
            x = graph.x.to(DEVICE).float()
            edge_index = graph.edge_index.to(DEVICE)
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(DEVICE)
            
            gnn_out = GNN_MODEL(x, edge_index, batch)
            gnn_prob = torch.softmax(gnn_out, dim=1)[0, 1].item()

        # RF & IF
        rf_prob = RF_MODEL.predict_proba(X_tab)[0, 1]
        
        if_model = IF_PACKAGE['model']
        calib = IF_PACKAGE['calibration']
        if_score = if_model.decision_function(X_tab)[0]
        if_prob = np.clip((calib['max'] - if_score) / (calib['max'] - calib['min']), 0.0, 1.0)

        # Stacking
        X_meta = np.array([[gnn_prob, rf_prob, if_prob]])
        final_prob = STACKING_MODEL.predict_proba(X_meta)[0, 1]
        
        result = "Attack" if final_prob > 0.5 else "Benign"
        
        return result

    except Exception as e:
        print(f"WORKER ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"