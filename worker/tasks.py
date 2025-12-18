import os
import torch
import joblib
import numpy as np
import sys

sys.path.append('/app')

from src.graph_builder import build_graph_from_log
from src.vector_builder import build_vector_from_log 
from src.model import GNNClassifier
from src.features import FEATURE_DIM 

MODEL_DIR = "/app/models"
DEVICE = torch.device('cpu')

def load_gnn_model(path, feature_dim, device):
    model = GNNClassifier(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# Global initialization
print("!!! WORKER INITIALIZING - LOADING MODELS !!!")
RF_MODEL = joblib.load(os.path.join(MODEL_DIR, 'combined_rf_model.pkl'))
IF_PACKAGE = joblib.load(os.path.join(MODEL_DIR, 'combined_if_model_calibrated.pkl'))
GNN_MODEL = load_gnn_model(os.path.join(MODEL_DIR, 'final_combined_gnn_model.pth'), FEATURE_DIM, DEVICE)
STACKING_MODEL = joblib.load(os.path.join(MODEL_DIR, 'final_stacking_model.pkl'))

def process_request(full_log):
    """
    Processes a full API log dictionary.
    """
    try:
        # Extract parts from the log structure
        request_part = full_log.get('request', {})
        response_part = full_log.get('response', {"status_code": 200, "body": ""})

        # Re-format for builders (handling url/path and empty bodies)
        formatted_request = {
            "method": request_part.get('method', 'GET'),
            "url": request_part.get('url', request_part.get('path', '/')),
            "headers": request_part.get('headers', {}),
            "body": request_part.get('body', '')
        }

        # 1. Feature Extraction
        graph = build_graph_from_log(formatted_request, response_part, label=0)
        gfv = build_vector_from_log(formatted_request, response_part)
        X_tab = np.array([gfv])

        # 2. GNN Inference
        with torch.no_grad():
            graph = graph.to(DEVICE)
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long).to(DEVICE)
            out = GNN_MODEL(graph.x, graph.edge_index, batch)
            gnn_prob = torch.exp(out)[0, 1].item()

        # 3. RF Inference
        rf_prob = RF_MODEL.predict_proba(X_tab)[0, 1]
        
        # 4. IF Calibration
        model_if = IF_PACKAGE['model']
        calib = IF_PACKAGE['calibration']
        if_score = model_if.decision_function(X_tab)[0]
        if_prob = np.clip((calib['max'] - if_score) / (calib['max'] - calib['min']), 0.0, 1.0)

        # 5. Stacking Decision
        X_meta = np.array([[gnn_prob, rf_prob, if_prob]])
        final_prob = STACKING_MODEL.predict_proba(X_meta)[0, 1]
        
        print(f"DEBUG: GNN:{gnn_prob:.3f}, RF:{rf_prob:.3f}, IF:{if_prob:.3f} -> FINAL:{final_prob:.3f}")
        return "Attack" if final_prob > 0.5 else "Benign"

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return f"Error: {str(e)}"