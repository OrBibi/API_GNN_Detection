import os
import sys
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, recall_score
from rq import get_current_job

# --- PATH CONFIGURATION ---
base_path = os.path.abspath("/app")
src_path = os.path.join(base_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import GNN model only (Stacking removed based on previous request)
from gnn_model import GNNClassifier 
from graph_builder import build_graph_from_log
from features import FEATURE_DIM

# --- CONFIGURATION ---
DEVICE = torch.device('cpu')
MODEL_PATH = os.path.join(base_path, "models", "final_combined_gnn_model.pth")
OUTPUT_DIR = os.path.join(base_path, "backend", "static", "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_confusion_matrix_image(y_true, y_pred, output_path):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    plt.title('GNN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_parquet_task(file_path):
    """
    RQ Task with Two-Stage Progress Reporting:
    1. Graph Construction Phase
    2. Inference Phase
    """
    try:
        job = get_current_job()
        
        if not os.path.isabs(file_path):
            file_path = os.path.join(base_path, file_path)

        # 1. Load Data
        df = pd.read_parquet(file_path)
        total_samples = len(df)

        # Initial Status Update
        if job:
            job.meta['total_samples'] = total_samples
            job.meta['progress'] = 0
            job.meta['stage'] = "Initializing..."
            job.save_meta()

        if not {'request', 'response'}.issubset(df.columns):
            return {'status': 'failed', 'error': 'Missing request/response columns'}

        has_label = 'label' in df.columns

        # 2. Load Model
        model = GNNClassifier(feature_dim=FEATURE_DIM).to(DEVICE)
        
        if not os.path.exists(MODEL_PATH):
            return {'status': 'failed', 'error': f'Model file missing at {MODEL_PATH}'}
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        # 3. PHASE A: Build Graphs (Now with Progress)
        graphs = []
        
        # We iterate with index to calculate progress
        for i, (_, row) in enumerate(df.iterrows()):
            lbl = row['label'] if has_label else 0
            graph = build_graph_from_log(row['request'], row['response'], lbl)
            graphs.append(graph)
            
            # Update progress every 1% or at least every 10 rows to reduce Redis overhead
            if job and (i % 10 == 0 or i == total_samples - 1):
                progress = int(((i + 1) / total_samples) * 100)
                job.meta['progress'] = progress
                job.meta['stage'] = "Building Graphs..."
                job.save_meta()

        # 4. PHASE B: Inference Loop (With Progress)
        BATCH_SIZE = 64
        loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
        predictions = []
        total_batches = len(loader)

        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1).cpu().tolist()
                predictions.extend(pred)
                
                # Update Progress
                if job:
                    progress = int(((i + 1) / total_batches) * 100)
                    job.meta['progress'] = progress
                    job.meta['stage'] = "Running Inference..."
                    job.save_meta()

        # 5. Save Output
        df['predict'] = predictions
        
        base_name = os.path.basename(file_path).replace('.parquet', '')
        res_filename = f"{base_name}_analyzed.parquet"
        res_path = os.path.join(OUTPUT_DIR, res_filename)
        
        df.to_parquet(res_path, index=False)
        
        result_data = {
            'status': 'completed',
            'download_url': f"/download/{res_filename}",
            'image_url': None,
            'accuracy': None,
            'attack_recall': None,
            'benign_recall': None,
            'total_samples': total_samples
        }

        # 6. Handle Metrics
        if has_label:
            img_filename = f"{base_name}_cm.png"
            img_path = os.path.join(OUTPUT_DIR, img_filename)
            generate_confusion_matrix_image(df['label'], df['predict'], img_path)
            
            acc = (df['label'] == df['predict']).mean()
            attack_rec = recall_score(df['label'], df['predict'], pos_label=1, zero_division=0)
            benign_rec = recall_score(df['label'], df['predict'], pos_label=0, zero_division=0)
            
            result_data['image_url'] = f"/static/results/{img_filename}"
            result_data['accuracy'] = acc
            result_data['attack_recall'] = attack_rec
            result_data['benign_recall'] = benign_rec

        return result_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}