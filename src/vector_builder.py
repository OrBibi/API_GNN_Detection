# src/vector_builder.py
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

# Import feature extraction and aggregation logic
from features import extract_node_features, aggregate_node_features, FEATURE_DIM

# --- Global State Replication ---
# Local state to build the node map for feature extraction without coupling to graph_builder.py

# Format: {node_id: {'type': str, 'key_or_value': str, 'raw_data': Any}}
_NODE_DETAILS_MAP: Dict[int, Dict[str, Any]] = {}
_NODE_ID_COUNTER: int = 0
_EDGES_LIST: List[Tuple[int, int]] = [] # Kept for structural consistency

def _reset_global_state():
    """Resets global state variables before processing a new log entry."""
    global _NODE_ID_COUNTER, _NODE_DETAILS_MAP, _EDGES_LIST
    _NODE_ID_COUNTER = 0
    _NODE_DETAILS_MAP = {}
    _EDGES_LIST = []

def _create_node_vector(node_type: str, key_or_value: str, raw_data: Any) -> int:
    """
    Creates a new node for feature extraction purposes and assigns a unique ID.
    """
    global _NODE_ID_COUNTER, _NODE_DETAILS_MAP
    
    current_id = _NODE_ID_COUNTER
    _NODE_ID_COUNTER += 1
    
    # Store the node's context and raw data (used by extract_node_features)
    _NODE_DETAILS_MAP[current_id] = {
        'type': node_type,
        'key_or_value': key_or_value,
        'raw_data': raw_data
    }
    return current_id


def _traverse_json_vector(data: Any, parent_id: int):
    """
    Recursively traverses a JSON structure using sorted keys to ensure deterministic 
    node ID assignment and feature extraction order.
    """
    global _EDGES_LIST
    
    if isinstance(data, dict):
        # Create a node representing the dictionary object
        dict_node_id = _create_node_vector(
            node_type='dict_container', 
            key_or_value='<OBJECT>', 
            raw_data=None
        )
        _EDGES_LIST.append((parent_id, dict_node_id))
        
        # Process each key-value pair using sorted keys for determinism
        for key, value in sorted(data.items()):
            
            # 1. Create a node for the KEY
            key_node_id = _create_node_vector(
                node_type='key', 
                key_or_value=str(key), 
                raw_data=key
            )
            _EDGES_LIST.append((dict_node_id, key_node_id))
            
            # 2. Recurse for the VALUE
            if isinstance(value, (dict, list)):
                _traverse_json_vector(value, key_node_id) 
            else:
                # If the value is primitive, create a VALUE node
                value_node_id = _create_node_vector(
                    node_type='primitive_value', 
                    key_or_value=str(value), 
                    raw_data=value
                )
                _EDGES_LIST.append((key_node_id, value_node_id))
                
    elif isinstance(data, list):
        # Create a node representing the list container
        list_node_id = _create_node_vector(
            node_type='list_container', 
            key_or_value='<ARRAY>', 
            raw_data=None
        )
        _EDGES_LIST.append((parent_id, list_node_id))
        
        # Process each item in the list
        for item in data:
            if isinstance(item, (dict, list)):
                _traverse_json_vector(item, list_node_id) 
            else:
                # If the item is primitive, create a VALUE node
                item_node_id = _create_node_vector(
                    node_type='primitive_value', 
                    key_or_value=str(item), 
                    raw_data=item
                )
                _EDGES_LIST.append((list_node_id, item_node_id))

# --- Main Vector Construction Function ---

def build_vector_from_log(request_data: Dict[str, Any], response_data: Dict[str, Any]) -> np.ndarray:
    """
    Constructs the Graph Feature Vector (GFV) deterministically from an API log entry.
    """
    
    _reset_global_state()
    
    # 1. Create Root Structure
    root_api_id = _create_node_vector(node_type='root_api', key_or_value='API_LOG', raw_data=None)

    # Request Root Node
    request_root_id = _create_node_vector(node_type='root_request', key_or_value='REQUEST_ROOT', raw_data=None)
    _EDGES_LIST.append((root_api_id, request_root_id))
    
    # Response Root Node
    response_root_id = _create_node_vector(node_type='root_response', key_or_value='RESPONSE_ROOT', raw_data=None)
    _EDGES_LIST.append((root_api_id, response_root_id))
    
    # 2. Traverse structures deterministically
    _traverse_json_vector(request_data, request_root_id)
    _traverse_json_vector(response_data, response_root_id)
    
    # 3. Handle Empty Case
    if _NODE_ID_COUNTER == 0:
        return np.zeros(FEATURE_DIM * 2, dtype=np.float32)

    # 4. Calculate node-level features (X) using the deterministic map
    x = extract_node_features(_NODE_DETAILS_MAP)
    
    # 5. Aggregate to final 1x132 vector (Mean and StdDev)
    gfv = aggregate_node_features(x)
    
    return gfv