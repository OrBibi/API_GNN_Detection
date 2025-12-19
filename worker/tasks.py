import os
import torch
import joblib
import numpy as np
import sys
import random

# Ensure the root directory is in the path for Docker structure
sys.path.append('/app')

from src.graph_builder import build_graph_from_log
from src.vector_builder import build_vector_from_log 
from src.model import GNNClassifier
from src.features import FEATURE_DIM 

# --- 1. Absolute Determinism Setup ---
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1) # Ensure single-threaded CPU ops for consistency
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ANSI Colors for Terminal Output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

MODEL_DIR = "/app/models"
DEVICE = torch.device('cpu') # Always CPU for production consistency

def load_gnn_model(path: str, feature_dim: int, device: torch.device):
    """Loads the pre-trained GNN and locks it in eval mode."""
    model = GNNClassifier(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    # Disable gradient calculations globally for this model
    for param in model.parameters():
        param.requires_grad = False
    return model

# --- Global Initialization (Loaded once during Worker startup) ---
print(f"{YELLOW}--- Initializing Malicious Detection Models ---{RESET}")
try:
    RF_MODEL = joblib.load(os.path.join(MODEL_DIR, 'combined_rf_model.pkl'))
    IF_PACKAGE = joblib.load(os.path.join(MODEL_DIR, 'combined_if_model_calibrated.pkl'))
    GNN_MODEL = load_gnn_model(os.path.join(MODEL_DIR, 'final_combined_gnn_model.pth'), FEATURE_DIM, DEVICE)
    STACKING_MODEL = joblib.load(os.path.join(MODEL_DIR, 'final_stacking_model.pkl'))
    print(f"{GREEN}[v] All models loaded successfully in EVAL mode.{RESET}")
except Exception as e:
    print(f"{RED}CRITICAL: Failed to load models: {e}{RESET}")
    sys.exit(1)

def process_request(full_log):
    """
    Main detection task. Processes a single log entry through the GNN-RF-IF ensemble.
    """
    try:
        # 1. Data Sanitization and Preparation
        req_raw = full_log.get('request', {})
        res_raw = full_log.get('response', {})
        full_url = req_raw.get('url', '')

        # Build consistent objects for the deterministic builders
        request_obj = {
            "method": req_raw.get('method', 'GET'),
            "url": full_url,
            "headers": req_raw.get('headers', {}),
            "body": req_raw.get('body', '')
        }

        response_obj = {
            "status_code": res_raw.get('status_code', 200),
            "headers": res_raw.get('headers', {}),
            "body": res_raw.get('body', '')
        }

        # 2. Deterministic Feature Building
        # The builders use MD5 and sorted keys internally as updated earlier
        graph = build_graph_from_log(request_obj, response_obj, label=0)
        gfv = build_vector_from_log(request_obj, response_obj)
        X_tab = np.array([gfv])

        # 3. Multi-Model Inference
        with torch.no_grad():
            graph = graph.to(DEVICE)
            # Create a batch of 1 for a single graph inference
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long).to(DEVICE)
            gnn_out = GNN_MODEL(graph.x.float(), graph.edge_index, batch)
            # Softmax [Benign, Attack] -> Index 1
            gnn_prob = torch.softmax(gnn_out, dim=1)[0, 1].item()

        # Random Forest Prediction
        rf_prob = RF_MODEL.predict_proba(X_tab)[0, 1]
        
        # Isolation Forest Calibration
        if_model = IF_PACKAGE['model']
        calib = IF_PACKAGE['calibration']
        if_score = if_model.decision_function(X_tab)[0]
        if_prob = np.clip((calib['max'] - if_score) / (calib['max'] - calib['min']), 0.0, 1.0)

        # 4. Final Stacking Decision
        X_meta = np.array([[gnn_prob, rf_prob, if_prob]])
        final_prob = STACKING_MODEL.predict_proba(X_meta)[0, 1]
        
        is_attack = final_prob > 0.5
        result_label = "ATTACK" if is_attack else "BENIGN"
        terminal_color = RED if is_attack else GREEN
        
        # --- Internal Logging ---
        print(f"\n{terminal_color}==========================================={RESET}")
        print(f"{terminal_color}DETECTION: {result_label}{RESET}")
        print(f"Final Ensemble Score: {final_prob:.4f}")
        print(f"Components -> GNN: {gnn_prob:.3f} | RF: {rf_prob:.3f} | IF: {if_prob:.3f}")
        print(f"URL: {full_url[:70]}...")
        print(f"{terminal_color}==========================================={RESET}")
        
        return {
            "prediction": result_label,
            "score": round(float(final_prob), 4),
            "details": {
                "gnn": round(float(gnn_prob), 3),
                "rf": round(float(rf_prob), 3),
                "if": round(float(if_prob), 3)
            }
        }

    except Exception as e:
        print(f"{RED}ERROR in Worker: {str(e)}{RESET}")
        return {"error": "Internal processing error during detection"}