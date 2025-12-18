# API Guard: AI-Powered HTTP Request Classifier

API Guard is an intelligent intrusion detection engine designed to classify **HTTP API Requests** as either "Benign" or "Attack" in real-time. The system leverages a **Stacking Ensemble** architecture, combining the structural analysis of Graph Neural Networks (GNN) with the statistical robustness of Random Forest and Isolation Forest models.

## 🚀 System Architecture

The system utilizes a distributed containerized environment:

* **FastAPI Backend:** Entry point for receiving HTTP logs and providing real-time status updates via persistent job IDs.
* **Redis Queue:** Acts as the message broker for asynchronous task distribution and processing.
* **ML Worker:** The computational engine that transforms raw HTTP data into graph and vector representations for multi-model inference.
* **Nginx Frontend:** A web dashboard for submitting requests and visualizing live threat detection results.

## 📁 Project Structure & Component Roles

### 1. Core Logic (`src/`)

* **`features.py`**: Defines the **66-dimensional** node feature vector. It handles one-hot encoding for node types and status codes, and calculates critical security ratios like URL encoding and special character frequency.
* **`graph_builder.py`**: Recursively traverses JSON/HTTP structures to build a graph representation where every key and value is a node, preserving the structural hierarchy of the API request.
* **`vector_builder.py`**: An optimized engine that extracts Graph Feature Vectors (GFV) without the overhead of full graph edge construction, used for traditional ML models.
* **`model.py`**: Defines the **GNNClassifier** architecture using GCN (Graph Convolutional Network) layers and global mean pooling for graph-level classification.
* **`data_loader.py`**: The entry point of the data pipeline. It extracts raw `7z` files, cleans labels to prevent leakage, and creates stratified train/validation/test splits in Parquet format.
* **`train.py`**: Orchestrates the training process for the GNN model, including data batching via PyTorch Geometric Loaders.
* **`train_random_forest.py`**: Trains the Random Forest classifier on tabular Graph Feature Vectors (GFVs).
* **`train_isolation_forest.py`**: Trains and calibrates the Isolation Forest for unsupervised anomaly detection using a specific min-max score normalization.
* **`train_stacking.py`**: Trains the final **Meta-Classifier** (Logistic Regression) which learns how to best weight the predictions from all sub-models.
* **`evaluate_models.py`**: Generates comprehensive performance reports and metrics for each individual model and the final ensemble.

### 2. Infrastructure & Deployment

* **`backend/`**: Contains `main.py`, the FastAPI server logic that handles the `/analyze` and `/status` endpoints.
* **`worker/`**: Contains `tasks.py`, the worker logic that loads models into memory and processes the task queue.
* **`frontend/`**: Contains the Nginx web dashboard files.
* **`models/`**: Storage for pre-trained weights (`.pth` for GNN, `.pkl` for RF/IF/Stacking meta-learner).
* **`data/`**: Divided into `raw` (source 7z files) and `processed` (final Parquet splits in the `splits/` subdirectory).
* **`reports/`**: Visualizations, figures, and confusion matrices generated during evaluation.
* **`Dockerfile`**: Defines the Python 3.9 environment, dependencies, and explicit copying of package init files to ensure correct module resolution.
* **`docker-compose.yml`**: Orchestrates the microservices, including the Redis broker and shared volumes for the `/app` root and `/models`.

## 🛠 Setup and Deployment

### Step 1: Data Initialization & Extraction

1. Place your raw `.7z` dataset files into the `data/raw/` directory.
2. Run the data loader to extract, label, and split the data into Parquet format:
```bash
python src/data_loader.py

```



### Step 2: Training the Machine Learning Pipeline

To generate the model weights found in the `models/` directory, run the following scripts in sequence:

```bash
# Train base classifiers
python src/train.py                  # GNN Model
python src/train_random_forest.py    # Random Forest
python src/train_isolation_forest.py # Isolation Forest

# Train the final Stacking Manager
python src/train_stacking.py

```

### Step 3: Performance Evaluation

After training, verify the individual and ensemble results using the test split:

```bash
python src/evaluate_models.py

```

### Step 4: Full System Launch (Docker)

Deploy the real-time detection stack:

```bash
# Clean previous state and start all containers
docker-compose down --volumes
docker-compose up --build

```

* **Dashboard**: `http://localhost`
* **API Documentation**: `http://localhost:8000/docs`

## 🔍 Real-Time Inference Process

1. **API Ingestion**: The API validates the request and enqueues a processing task into Redis.
2. **Feature Engineering**: The Worker reconstructs the raw HTTP format and builds both a graph and a tabular feature vector (66-dim).
3. **Parallel Inference**: The GNN analyzes structural anomalies, while the RF and IF analyze statistical outliers.
4. **Final Decision**: The Stacking Manager provides the final verdict based on the weighted confidence of all sub-models.

## 📊 Conclusions & Results

The evaluation demonstrates the power of the Stacking Ensemble approach compared to individual models:

| Model | Global Accuracy | Attack Detection (Recall) | False Alarm Rate (FPR) |
| --- | --- | --- | --- |
| **GNN Only** | 87.86% | **91.11%** | 12.60% |
| **Random Forest** | 92.26% | 46.64% | **1.34%** |
| **Isolation Forest** | 78.30% | 59.60% | 19.07% |
| **Stacking Ensemble** | **94.11%** | 85.78% | 4.72% |

### Summary

* **GNN** provides the highest raw detection rate (Recall) but suffers from a higher false alarm rate.
* **Random Forest** is extremely precise with the lowest false alarm rate but misses nearly half of the attack variations.
* **The Stacking Ensemble** (Final Result) achieves the best overall balance, reaching a **Global Accuracy of 94.11%**. It successfully combines the GNN's sensitivity with the Random Forest's precision, creating a robust production-ready classifier for HTTP API security.

---