import os
import json
import logging
import pandas as pd
import py7zr
import shutil
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

# --- Configuration ---
# Set up logging format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths and static SEED for reproducibility
DATA_RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits") # New directory for split results

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True) # Ensure the splits directory exists

RANDOM_SEED = 42  # Fixed seed for reproducible random splitting

# Define all datasets to process
DATASETS_TO_PROCESS = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4']


# --- Helper Functions (Extraction and Cleaning) ---

def extract_7z_files(zip_file_name: str, output_subdir: str) -> str:
    """
    Extracts a 7z file to a specified directory and returns the path to the extracted JSON file.
    """
    zip_file_path = os.path.join(DATA_RAW_DIR, zip_file_name)
    output_dir = os.path.join(PROCESSED_DIR, output_subdir)
    
    # Clean up previous extraction attempt
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Extracting {zip_file_name} from {DATA_RAW_DIR} to {output_dir}...")
    
    try:
        with py7zr.SevenZipFile(zip_file_path, mode='r') as archive:
            archive.extractall(path=output_dir)
            
        json_filename = [f for f in os.listdir(output_dir) if f.endswith('.json')]
            
        if not json_filename:
            logging.error(f"Error: No JSON file found in {output_dir}")
            return None
                
        json_file_path = os.path.join(output_dir, json_filename[0])
        logging.info(f"Extraction successful. JSON file path: {json_file_path}")
        return json_file_path
        
    except FileNotFoundError:
        logging.error(f"Error: 7z file not found at {zip_file_path}. Please check your data/raw directory.")
        return None
    except Exception as e:
        logging.error(f"Error during 7z extraction: {e}")
        return None


def extract_label_from_request(request_dict):
    """
    Derives the binary label (1/0) from the nested 'Attack_Tag' key.
    """
    if isinstance(request_dict, dict) and 'Attack_Tag' in request_dict:
        return 1
    return 0


def clean_request_tag(request_dict):
    """
    Removes the 'Attack_Tag' key from the request dictionary to prevent label leakage.
    """
    if isinstance(request_dict, dict):
        request_dict.pop('Attack_Tag', None)
    return request_dict


# --- Splitting Logic ---

def get_split_ratios(dataset_name: str) -> Tuple[float, float]:
    """
    Returns (test_size, val_size) based on the dataset name.
    
    Dataset 1 (Small/Balanced): 80% Train / 10% Val / 10% Test
    Others (Large/Imbalanced): 70% Train / 10% Val / 20% Test
    """
    # 80/10/10 split
    if dataset_name == 'dataset_1':
        return (0.1, 0.1)  # (test_size, val_size)
        
    # 70/10/20 split
    else: 
        return (0.2, 0.1)  # (test_size, val_size)


def split_data(df: pd.DataFrame, ds_name: str, random_state: int = RANDOM_SEED) -> Tuple:
    """
    Splits the data into Train, Validation, and Test sets based on dynamic ratios.
    The split is stratified by label to maintain the class distribution in all sets.
    """
    test_size, val_size = get_split_ratios(ds_name)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 1. First split: Separate the Test Set
    # X_temp will contain the remaining data for Train/Validation
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Second split: Separate the Validation Set from X_temp
    # Calculate the ratio of Validation set relative to X_temp
    val_ratio_in_temp = val_size / (1.0 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_in_temp, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# --- Main Processing Logic ---

def process_single_dataset(dataset_number: str) -> pd.DataFrame:
    """
    Loads, combines, and cleans a single dataset, ensuring robustness to file format.
    """
    logging.info(f"Starting processing for {dataset_number}...")
    
    train_zip_name = f"{dataset_number}_train.7z"
    val_zip_name = f"{dataset_number}_val.7z"
    
    train_path = extract_7z_files(train_zip_name, f"{dataset_number}_train_extracted")
    val_path = extract_7z_files(val_zip_name, f"{dataset_number}_val_extracted")

    if not train_path or not val_path:
        raise FileNotFoundError(f"One or both required JSON files could not be extracted for {dataset_number}.")
        
    # 3. Load DataFrames: Try to load as a single JSON array (full load)
    logging.info(f"Attempting to load JSON files for {dataset_number} as a single JSON array...")
    try:
        train_df = pd.read_json(train_path) 
        val_df = pd.read_json(val_path)
        logging.info("JSON files loaded successfully as a single array.")
    except Exception as e:
        # Fallback to line-by-line reading for different JSON structures
        logging.warning(f"Failed to read JSON files for {dataset_number} as a single array: {e}. Falling back to line-by-line method (lines=True).")
        train_df = pd.read_json(train_path, lines=True) 
        val_df = pd.read_json(val_path, lines=True) 

    # 4. Combine DataFrames
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # 5. Ensure the columns are named 'request' and 'response'
    required_cols = ['request', 'response']
    
    # Handling numerical columns (if fallback to lines=True occurred)
    if all(isinstance(col, int) for col in combined_df.columns):
        if len(combined_df.columns) >= 2:
            current_cols = combined_df.columns.tolist()
            combined_df = combined_df.rename(columns={current_cols[0]: 'request', current_cols[1]: 'response'})
            cols_to_drop = [col for col in combined_df.columns if isinstance(col, int)]
            combined_df = combined_df.drop(columns=cols_to_drop, errors='ignore')
    
    if not all(col in combined_df.columns for col in required_cols):
        logging.error(f"Data structure mismatch in {dataset_number}. Expected columns {required_cols} not found.")
        raise KeyError(f"'request' or 'response' column missing in {dataset_number}.")
    
    combined_df = combined_df[required_cols]

    # 6. Extract the label and clean the 'request' column
    combined_df['label'] = combined_df['request'].apply(extract_label_from_request)
    combined_df['request'] = combined_df['request'].apply(clean_request_tag)
    
    # 7. Final Data Check
    label_counts = combined_df['label'].value_counts().to_dict()
    logging.info(f"Processing successful for {dataset_number}. Total records: {len(combined_df)}. Counts: {label_counts}")
    return combined_df

def load_processed_data(dataset_name: str, split_name: str) -> pd.DataFrame:
    """
    Loads a processed Parquet split (train/val/test) for a given dataset 
    from the data/processed/splits directory.
    """
    # Assuming SPLITS_DIR is defined globally above
    file_path = os.path.join(SPLITS_DIR, f"{dataset_name}_{split_name}.parquet")
    
    if not os.path.exists(file_path):
        logging.error(f"FATAL: Processed data file not found at: {file_path}. Please run data_loader.py first.")
        raise FileNotFoundError(f"Processed data file not found at: {file_path}")
    
    # Load the DataFrame from Parquet
    df = pd.read_parquet(file_path)
    logging.info(f"Loaded {split_name} data for {dataset_name}. Shape: {df.shape}")
    return df


# --- Main Execution ---

if __name__ == "__main__":
    
    all_processed_data: Dict[str, pd.DataFrame] = {}
    
    # 1. LOAD AND PROCESS ALL DATASETS
    for ds_name in DATASETS_TO_PROCESS:
        try:
            combined_data = process_single_dataset(ds_name)
            all_processed_data[ds_name] = combined_data
        
        except Exception as e:
            logging.error(f"An error occurred while processing {ds_name}: {e}")
            
    logging.info("-------------------------------------------")
    logging.info("--- Data Splitting Summary (Stratified) ---")
    logging.info(f"Using fixed random state: {RANDOM_SEED} for reproducibility.")
    logging.info("-------------------------------------------")
    
    # 2. SPLIT AND SAVE ALL DATASETS DYNAMICALLY
    
    if all_processed_data:
        for ds_name, df in all_processed_data.items():
            
            test_size, val_size = get_split_ratios(ds_name)
            
            try:
                # Perform the split
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, ds_name=ds_name, random_state=RANDOM_SEED)
                
                # Report the split sizes and ratios used
                train_ratio = 1.0 - test_size - val_size
                split_ratio_str = f"{int(train_ratio*100)}/{int(val_size*100)}/{int(test_size*100)}"
                
                logging.info(f"Dataset {ds_name} Split ({split_ratio_str}):")
                logging.info(f"  Train: {len(X_train)} rows")
                logging.info(f"  Validation: {len(X_val)} rows")
                logging.info(f"  Test: {len(X_test)} rows")
                
                # 3. SAVE THE SPLIT DATA TO DISK
                base_path = os.path.join(SPLITS_DIR, ds_name)
                
                # Combine X (features) and y (label) into one DataFrame for each set for easy saving and reloading
                train_df_to_save = pd.concat([X_train, y_train], axis=1)
                val_df_to_save = pd.concat([X_val, y_val], axis=1)
                test_df_to_save = pd.concat([X_test, y_test], axis=1)
                
                # Save using Parquet format
                train_df_to_save.to_parquet(f"{base_path}_train.parquet", index=False)
                val_df_to_save.to_parquet(f"{base_path}_val.parquet", index=False)
                test_df_to_save.to_parquet(f"{base_path}_test.parquet", index=False)
                
                logging.info(f"  --> Successfully saved splits to {SPLITS_DIR} in Parquet format.")
                
            except ValueError as e:
                logging.error(f"Could not split {ds_name} into 3 sets with Stratification. Error: {e}. Skipping split for this dataset.")

    logging.info("-------------------------------------------")
    logging.info("Data loading, preparation, and saving completed.")