import os
import torch
import joblib
import numpy as np
import sys

# Ensure the root is in path
sys.path.append('/app')

print("!!! WORKER IS INITIALIZING AND LOADING MODELS !!!")

# English comments as requested
# Import logic from the parallel 'src' folder
from src.graph_builder import build_graph_from_log
from src.vector_builder import build_vector_from_log 
from src.model import GNNClassifier
from src.features import FEATURE_DIM 

# Path to the models folder
MODEL_DIR = "/app/models"
DEVICE = torch.device('cpu')

def load_gnn_model(path, feature_dim, device):
    """Loads the pre-trained GNN model from disk."""
    model = GNNClassifier(feature_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def get_calibrated_if_probs(if_package, X_vec):
    """Applies calibration logic to Isolation Forest scores as seen in evaluation."""
    model = if_package['model']
    calib = if_package['calibration']
    global_min, global_max = calib['min'], calib['max']
    scores = model.decision_function(X_vec)
    probs = (global_max - scores) / (global_max - global_min)
    return np.clip(probs, 0.0, 1.0)

# --- Global Model Initialization ---
# Models are loaded once at worker startup to ensure fast inference
print("Loading ensemble models from /app/models...")
RF_MODEL = joblib.load(os.path.join(MODEL_DIR, 'combined_rf_model.pkl'))
IF_PACKAGE = joblib.load(os.path.join(MODEL_DIR, 'combined_if_model_calibrated.pkl'))
GNN_MODEL = load_gnn_model(os.path.join(MODEL_DIR, 'final_combined_gnn_model.pth'), FEATURE_DIM, DEVICE)
STACKING_MODEL = joblib.load(os.path.join(MODEL_DIR, 'final_stacking_model.pkl'))
print("Ensemble models loaded and ready for inference.")

def process_request(request_data):
    """
    Worker Task: Processes incoming API request data and returns a classification result.
    """
    try:
        # 1. Data Preparation
        method = request_data.get('method', 'GET')
        path = request_data.get('path', '/')
        body = request_data.get('body', '')
        
        # Reconstruct raw format expected by the graph and vector builders
        raw_request = f"{method} {path} HTTP/1.1\r\n\r\n{body}"
        raw_response = "" # WAF usually inspects requests before a response is generated
        
        # Generate graph data and tabular features
        graph = build_graph_from_log(raw_request, raw_response, label=0)
        gfv = build_vector_from_log(raw_request, raw_response)
        
        # Reshape tabular features to (1, FEATURE_DIM) for Scikit-Learn compatibility
        X_tab = np.array([gfv]) 

        # 2. Multi-Model Inference
        
        # A. GNN Inference
        with torch.no_grad():
            graph = graph.to(DEVICE)
            # Pass node features, edge index, and a single-batch index tensor
            out = GNN_MODEL(graph.x, graph.edge_index, torch.tensor([0])) 
            gnn_prob = torch.exp(out)[0, 1].item()

        # B. Random Forest Inference
        rf_prob = RF_MODEL.predict_proba(X_tab)[0, 1]

        # C. Isolation Forest Inference (Calibrated)
        if_prob = get_calibrated_if_probs(IF_PACKAGE, X_tab)[0]

        # 3. Final Stacking Decision
        # Combine probabilities from base models into a meta-feature vector
        X_meta = np.array([[gnn_prob, rf_prob, if_prob]])
        final_prob = STACKING_MODEL.predict_proba(X_meta)[0, 1]
        
        # Classification threshold set at 0.5
        prediction = "Attack" if final_prob > 0.5 else "Benign"

        return {
            "prediction": prediction,
            "confidence": round(float(final_prob), 4),
            "details": {
                "gnn_score": round(gnn_prob, 4),
                "rf_score": round(rf_prob, 4),
                "if_score": round(if_prob, 4)
            }
        }

    except Exception as e:
        # Error logging for debugging within Docker logs
        return {"prediction": "Error", "message": str(e)}