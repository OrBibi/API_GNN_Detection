# src/vector_builder.py
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

# We must import the feature aggregation logic
from features import extract_node_features, aggregate_node_features, FEATURE_DIM

# --- Global State Replication ---
# We replicate the state management logic from graph_builder.py 
# to build the node map without coupling to the original file's globals.

# Format: {node_id: {'type': str, 'key_or_value': str, 'raw_data': Any}}
_NODE_DETAILS_MAP: Dict[int, Dict[str, Any]] = {}
_NODE_ID_COUNTER: int = 0
_EDGES_LIST: List[Tuple[int, int]] = [] # Edges are ignored but kept for completeness

def _reset_global_state():
    """Resets global state variables before processing a new log."""
    global _NODE_ID_COUNTER, _NODE_DETAILS_MAP, _EDGES_LIST
    _NODE_ID_COUNTER = 0
    _NODE_DETAILS_MAP = {}
    _EDGES_LIST = []

def _create_node_vector(node_type: str, key_or_value: str, raw_data: Any) -> int:
    """
    Creates a new node (for feature extraction purposes), assigns a unique ID, 
    and stores its details in the local map.
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
    Recursively traverses a JSON structure (dict or list) to build nodes
    but *only* for the purpose of feature extraction. Edge logic is minimal.
    """
    global _EDGES_LIST
    
    if isinstance(data, dict):
        # Create a node representing the dictionary object itself
        dict_node_id = _create_node_vector(
            node_type='dict_container', 
            key_or_value='<OBJECT>', 
            raw_data=None
        )
        _EDGES_LIST.append((parent_id, dict_node_id)) # Still track the edge for completeness
        
        # Process each key-value pair
        for key, value in data.items():
            
            # 1. Create a node for the KEY
            key_node_id = _create_node_vector(
                node_type='key', 
                key_or_value=key, 
                raw_data=key
            )
            _EDGES_LIST.append((dict_node_id, key_node_id))
            
            # 2. Recurse for the VALUE, with the Key Node as the parent
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
        # Create a node representing the list/array object itself
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
    Main function to construct the Graph Feature Vector (GFV) from an API log entry.
    This bypasses graph edge construction entirely for efficiency.
    """
    
    _reset_global_state()
    
    # 1. Create Root Nodes
    
    # The ultimate root node (for the entire API log)
    root_api_id = _create_node_vector(node_type='root_api', key_or_value='API_LOG', raw_data=None)

    # Request Root Node
    request_root_id = _create_node_vector(node_type='root_request', key_or_value='REQUEST_ROOT', raw_data=None)
    _EDGES_LIST.append((root_api_id, request_root_id))
    
    # Response Root Node
    response_root_id = _create_node_vector(node_type='root_response', key_or_value='RESPONSE_ROOT', raw_data=None)
    _EDGES_LIST.append((root_api_id, response_root_id))
    
    
    # 2. Traverse Request and Response structures
    _traverse_json_vector(request_data, request_root_id)
    _traverse_json_vector(response_data, response_root_id)
    
    
    # 3. Handle Empty Case
    if _NODE_ID_COUNTER == 0:
        return np.zeros(FEATURE_DIM * 2, dtype=np.float32)

    # 4. CALCULATE NODE FEATURES (X)
    # Uses the map created by the local traversal logic
    x = extract_node_features(_NODE_DETAILS_MAP)
    
    
    # 5. AGGREGATE TO FINAL GFV (1x132 vector)
    gfv = aggregate_node_features(x)
    
    return gfv