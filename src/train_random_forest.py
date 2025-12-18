import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from typing import Tuple 
from tqdm import tqdm

# Imports from your project structure
from vector_builder import build_vector_from_log
from data_loader import load_processed_data, SPLITS_DIR 

# --- Configuration ---
DATASETS_TO_COMBINE = ['dataset_1','dataset_2', 'dataset_3', 'dataset_4'] 
MODEL_OUTPUT_DIR = 'models'
FINAL_MODEL_NAME = 'combined_rf_model.pkl' 
RF_MODEL = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def extract_gfv_pipeline(df: pd.DataFrame, ds_name: str, split_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts Graph Feature Vectors (GFV) and labels."""
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

def train_rf_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Trains on Train set, Evaluates on Val set."""
    
    print("\n--- Training Random Forest (Combined Data) ---")
    
    rf_model = RF_MODEL
    
    # 1. Train ONLY on Training Data (No Leakage)
    print(f"INFO: Fitting model on {len(X_train)} training samples...")
    rf_model.fit(X_train, y_train)
    
    # 2. Save Model
    final_path = os.path.join(MODEL_OUTPUT_DIR, FINAL_MODEL_NAME)
    joblib.dump(rf_model, final_path)
    print(f"SUCCESS: Random Forest model saved to {final_path}")

    # 3. Predict on Validation Data (For Reporting Only)
    print("INFO: Evaluating on Validation Set...")
    y_pred_val = rf_model.predict(X_val)
    
    # 4. Print Comparison Metrics
    val_acc = accuracy_score(y_val, y_pred_val)
    val_rec = recall_score(y_val, y_pred_val, pos_label=1)
    val_f1 = f1_score(y_val, y_pred_val, pos_label=1)
    
    print("\n" + "="*60)
    print(f"RESULTS: RANDOM FOREST (On Validation Set)")
    print("="*60)
    print(f"Val Acc:    {val_acc:.4f}")
    print(f"Val Recall: {val_rec:.4f}")
    print(f"Val F1:     {val_f1:.4f}")
    print("-" * 60)
    # print(classification_report(y_val, y_pred_val, digits=4)) # Optional

if __name__ == '__main__':
    print(f"INFO: Starting RF Process on: {DATASETS_TO_COMBINE}...")
    
    all_gfv_train, all_labels_train = [], []
    all_gfv_val, all_labels_val = [], []

    for ds_name in DATASETS_TO_COMBINE:
        try:
            # Load BOTH splits
            df_train = load_processed_data(ds_name, 'train')
            df_val = load_processed_data(ds_name, 'val')
        except Exception as e:
            print(f"WARNING: Skipping {ds_name}: {e}")
            continue

        # Extract Train
        X_t, y_t = extract_gfv_pipeline(df_train, ds_name, "TRAIN")
        all_gfv_train.append(X_t)
        all_labels_train.append(y_t)
        
        # Extract Val
        X_v, y_v = extract_gfv_pipeline(df_val, ds_name, "VAL")
        all_gfv_val.append(X_v)
        all_labels_val.append(y_v)

    # Combine
    if not all_gfv_train: exit()
    
    X_final_train = np.concatenate(all_gfv_train, axis=0)
    y_final_train = np.concatenate(all_labels_train, axis=0)
    
    X_final_val = np.concatenate(all_gfv_val, axis=0)
    y_final_val = np.concatenate(all_labels_val, axis=0)
    
    train_rf_model(X_final_train, y_final_train, X_final_val, y_final_val)