import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm
from typing import Tuple

# Imports from your project structure
from vector_builder import build_vector_from_log
from data_loader import load_processed_data, SPLITS_DIR 

# --- Configuration ---
DATASETS_TO_COMBINE = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']
MODEL_OUTPUT_DIR = 'models'
FINAL_MODEL_NAME = 'combined_if_model_calibrated.pkl' 
IF_MODEL = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def extract_gfv_pipeline(df: pd.DataFrame, ds_name: str, split_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts Graph Feature Vectors (GFV)."""
    gfv_list = []
    labels = []
    desc = f"Extracting GFV ({ds_name} - {split_type})"
    for index, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            gfv = build_vector_from_log(row['request'], row['response']) 
            gfv_list.append(gfv)
            labels.append(row['label']) 
        except Exception:
            continue
    return np.array(gfv_list), np.array(labels)

def train_and_calibrate_if(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Trains on Train, Calibrates on Train, Evaluates on Val."""
    print("\n--- Training & Calibrating Isolation Forest ---")
    
    # 1. Train ONLY on Benign samples from TRAINING set
    X_if_train = X_train[y_train == 0] 
    print(f"INFO: Training IF model on {len(X_if_train)} benign samples (from Train)...")
    
    if_model = IF_MODEL
    if_model.fit(X_if_train) 
    
    # 2. Calibration: Find Min/Max scores based on TRAINING set distribution
    print("INFO: Calibrating scores on full TRAINING dataset...")
    train_scores = if_model.decision_function(X_train)
    global_max = train_scores.max()
    global_min = train_scores.min()
    print(f"CALIBRATION (from Train): Max={global_max:.4f}, Min={global_min:.4f}")
    
    # 3. Save Package
    model_package = {
        'model': if_model,
        'calibration': {'min': global_min, 'max': global_max}
    }
    final_path = os.path.join(MODEL_OUTPUT_DIR, FINAL_MODEL_NAME)
    joblib.dump(model_package, final_path)
    print(f"SUCCESS: Saved to {final_path}")
    
    # 4. Evaluate on VALIDATION Set (Using Train Calibration)
    print("INFO: Evaluating on Validation Set...")
    val_scores = if_model.decision_function(X_val)
    
    # Normalize using TRAIN calibration (Strict logic)
    probs = (global_max - val_scores) / (global_max - global_min)
    preds = (probs > 0.5).astype(int)
    
    val_acc = accuracy_score(y_val, preds)
    val_rec = recall_score(y_val, preds, pos_label=1)
    val_f1 = f1_score(y_val, preds, pos_label=1)

    print("\n" + "="*60)
    print(f"RESULTS: ISOLATION FOREST (On Validation Set)")
    print("="*60)
    print(f"Val Acc:    {val_acc:.4f}")
    print(f"Val Recall: {val_rec:.4f}")
    print(f"Val F1:     {val_f1:.4f}")
    print("-" * 60)

if __name__ == '__main__':
    print(f"INFO: Starting IF Process on: {DATASETS_TO_COMBINE}...")
    
    all_gfv_train, all_labels_train = [], []
    all_gfv_val, all_labels_val = [], []

    for ds_name in DATASETS_TO_COMBINE:
        try:
            df_train = load_processed_data(ds_name, 'train')
            df_val = load_processed_data(ds_name, 'val')
        except Exception as e:
            print(f"Warning: Skipping {ds_name}: {e}")
            continue

        X_t, y_t = extract_gfv_pipeline(df_train, ds_name, "TRAIN")
        all_gfv_train.append(X_t)
        all_labels_train.append(y_t)
        
        X_v, y_v = extract_gfv_pipeline(df_val, ds_name, "VAL")
        all_gfv_val.append(X_v)
        all_labels_val.append(y_v)

    if not all_gfv_train: exit()

    X_final_train = np.concatenate(all_gfv_train, axis=0)
    y_final_train = np.concatenate(all_labels_train, axis=0)
    
    X_final_val = np.concatenate(all_gfv_val, axis=0)
    y_final_val = np.concatenate(all_labels_val, axis=0)
    
    train_and_calibrate_if(X_final_train, y_final_train, X_final_val, y_final_val)