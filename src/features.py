# src/features.py
import torch
import numpy as np
import hashlib  
from typing import Dict, Any, List

# --- Configuration ---
# FEATURE_DIM is set to 66 to accommodate all fast features:
# 7 (Type OHE) + 4 (Text/Hash/Ratio) + 3 (Status Code OHE) + 1 (Key Length) + ~51 (Padding)
FEATURE_DIM = 66
EMBEDDER = None
print("INFO: Using 66-dim deterministic features (MD5 based hash).")

# --- Global Mappings ---

NODE_TYPE_MAPPING = {
    'root_api': 0, 
    'root_request': 1,
    'root_response': 2,
    'dict_container': 3,
    'list_container': 4,
    'key': 5,
    'primitive_value': 6
}

# Define special characters highly indicative of injection attacks.
# Use sorted list to ensure deterministic behavior.
_SPECIAL_CHARS = sorted(["'", ";", "<", ">", "=", "(", ")", "--", "%", "&"])

# --- Helper Functions ---

def _get_deterministic_hash(str_value: str) -> float:
    """
    Replaces built-in hash() which is randomized per session.
    Uses MD5 to produce a consistent float between 0 and 1.
    """
    if not str_value:
        return 0.0
    # Create a stable MD5 hash of the string
    hash_object = hashlib.md5(str_value.encode('utf-8'))
    # Take the first 8 characters of the hex digest and convert to int
    hash_int = int(hash_object.hexdigest()[:8], 16)
    # Normalize to [0, 1] range
    return (hash_int % 1000) / 1000.0

def _extract_status_code_features(status_code: Any) -> torch.Tensor:
    """Extracts features from the HTTP status code (3-dim one-hot)."""
    features = torch.zeros(3)
    try:
        code = int(status_code)
        if 200 <= code < 300: # Success (2xx)
            features[0] = 1.0
        elif 400 <= code < 500: # Client Error (4xx) - High suspicion
            features[1] = 1.0
        elif 500 <= code < 600: # Server Error (5xx)
            features[2] = 1.0
    except:
        pass 
    return features

def _extract_special_char_ratio(str_value: str) -> float:
    """Calculates the ratio of special attack-related characters in a string."""
    if not str_value:
        return 0.0
    
    count = 0
    # Simple, fast counting of character occurrences
    for char in str_value:
        if char in _SPECIAL_CHARS:
            count += 1
            
    # Normalize by the total length of the string
    return count / len(str_value)

def _extract_url_encoding_ratio(str_value: str) -> float:
    """
    Calculates the ratio of URL-encoded characters (%xx) in a string.
    """
    if not str_value or len(str_value) < 3:
        return 0.0
    
    encoded_count = 0
    i = 0
    while i < len(str_value) - 2:
        if str_value[i] == '%' and str_value[i+1:i+3].isalnum():
            encoded_count += 1
            i += 3 
        else:
            i += 1
            
    return (encoded_count * 3) / len(str_value)


# --- Main Feature Extractor ---

def extract_node_features(node_details_map: Dict[int, Dict[str, Any]]) -> torch.Tensor:
    """
    Calculates the feature vector X for all nodes using deterministic features (D=66).
    """
    
    num_nodes = len(node_details_map)
    if num_nodes == 0:
        return torch.empty(0, FEATURE_DIM)
        
    feature_matrix = torch.zeros((num_nodes, FEATURE_DIM), dtype=torch.float)
    
    # Sort keys to ensure deterministic processing order
    for node_id in sorted(node_details_map.keys()):
        details = node_details_map[node_id]
        node_type = details['type']
        raw_value = details['raw_data']
        key_or_value = details['key_or_value']
        
        # 1. Base Feature: Node Type (One-Hot on the first 7 dimensions)
        type_index = NODE_TYPE_MAPPING.get(node_type, 7)
        if type_index < 7:
            feature_matrix[node_id, type_index] = 1.0
            
        start_idx = 7 # Content features begin here
        
        if node_type == 'primitive_value':
            str_value = str(raw_value)
            
            # A. Simple Text Features (Indices 7-10)
            try:
                length_norm = min(len(str_value), 100) / 100.0   # Index 7 (Length)
                # Updated to use deterministic MD5 hash
                hash_value = _get_deterministic_hash(str_value)  # Index 8 (Hash)
                special_ratio = _extract_special_char_ratio(str_value) # Index 9 (Special Chars)
                url_ratio = _extract_url_encoding_ratio(str_value)    # Index 10 (URL Encoding)
                
                # Place Features
                feature_matrix[node_id, start_idx] = length_norm
                feature_matrix[node_id, start_idx + 1] = hash_value
                feature_matrix[node_id, start_idx + 2] = special_ratio
                feature_matrix[node_id, start_idx + 3] = url_ratio 
                    
            except:
                 pass
                 
            # B. Status Code Feature (Indices 11-13)
            status_idx = start_idx + 4 # Status code features start at Index 11
            if node_id > 0 and node_details_map.get(node_id - 1, {}).get('key_or_value') == 'status_code':
                 status_features = _extract_status_code_features(raw_value)
                 feature_matrix[node_id, status_idx : status_idx + 3] = status_features

        # 3. Simple Features for Keys (Index 14)
        elif node_type == 'key':
             key_length_norm = min(len(key_or_value), 30) / 30.0
             key_len_idx = start_idx + 7 # Key length index is 14
             if key_len_idx < FEATURE_DIM: 
                 feature_matrix[node_id, key_len_idx] = key_length_norm
                 
    return feature_matrix

def aggregate_node_features(node_features: torch.Tensor) -> np.ndarray:
    """
    Performs statistical aggregation (Mean and Standard Deviation) on X.
    """
    GFV_DIM = FEATURE_DIM * 2 
    
    if node_features.numel() == 0:
        return np.zeros(GFV_DIM, dtype=np.float32)

    mean_features = torch.mean(node_features, dim=0).numpy()
    
    if node_features.size(0) > 1:
        std_features = torch.std(node_features, dim=0).numpy()
    else:
        std_features = np.zeros(FEATURE_DIM, dtype=np.float32)

    gfv = np.concatenate([mean_features, std_features])
    return gfv