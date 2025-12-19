# src/graph_builder.py
import json
import torch
from torch_geometric.data import Data
from typing import Dict, Any, Tuple, List

# Import the feature extractor from the features module
from features import extract_node_features

# Global counters and structures for graph construction
node_id_counter = 0
node_details_map = {}
edges_list = []

def _create_node(node_type: str, key_or_value: str, raw_data: Any) -> int:
    """
    Creates a new node, assigns a unique ID, and stores its details.
    """
    global node_id_counter, node_details_map
    
    current_id = node_id_counter
    node_id_counter += 1
    
    # Store the node's context and raw data (used by extract_node_features)
    node_details_map[current_id] = {
        'type': node_type,
        'key_or_value': key_or_value,
        'raw_data': raw_data
    }
    return current_id

def _traverse_json(data: Any, parent_id: int):
    """
    Recursively traverses a JSON structure (dict or list) to build nodes and edges.
    Uses sorted keys for deterministic node ID assignment.
    """
    global edges_list
    
    if isinstance(data, dict):
        # Create a node representing the dictionary object itself
        dict_node_id = _create_node(
            node_type='dict_container', 
            key_or_value='<OBJECT>', 
            raw_data=None
        )
        edges_list.append((parent_id, dict_node_id))
        
        # Sorting keys ensures deterministic node ID assignment across runs
        for key, value in sorted(data.items()):
            
            # 1. Create a node for the KEY
            key_node_id = _create_node(
                node_type='key', 
                key_or_value=str(key), 
                raw_data=key
            )
            # Add edge: Dictionary Container -> Key Node
            edges_list.append((dict_node_id, key_node_id))
            
            # 2. Recurse for the VALUE
            if isinstance(value, (dict, list)):
                _traverse_json(value, key_node_id)
            else:
                # If the value is primitive, create a VALUE node
                value_node_id = _create_node(
                    node_type='primitive_value', 
                    key_or_value=str(value), 
                    raw_data=value
                )
                edges_list.append((key_node_id, value_node_id))
                
    elif isinstance(data, list):
        # Create a node representing the list/array object itself
        list_node_id = _create_node(
            node_type='list_container', 
            key_or_value='<ARRAY>', 
            raw_data=None
        )
        edges_list.append((parent_id, list_node_id))
        
        # Process each item in the list
        for item in data:
            if isinstance(item, (dict, list)):
                _traverse_json(item, list_node_id)
            else:
                item_node_id = _create_node(
                    node_type='primitive_value', 
                    key_or_value=str(item), 
                    raw_data=item
                )
                edges_list.append((list_node_id, item_node_id))

def build_graph_from_log(request_data: Dict[str, Any], response_data: Dict[str, Any], label: int) -> Data:
    """
    Main function to construct a single PyTorch Geometric Data object from an API log entry.
    """
    global node_id_counter, node_details_map, edges_list
    
    # 0. Reset global state for a new graph construction
    node_id_counter = 0
    node_details_map = {}
    edges_list = []
    
    # 1. Create Root Structure
    root_api_id = _create_node(node_type='root_api', key_or_value='API_LOG', raw_data=None)

    request_root_id = _create_node(node_type='root_request', key_or_value='REQUEST_ROOT', raw_data=None)
    edges_list.append((root_api_id, request_root_id))
    
    response_root_id = _create_node(node_type='root_response', key_or_value='RESPONSE_ROOT', raw_data=None)
    edges_list.append((root_api_id, response_root_id))
    
    # 2. Traverse Request and Response structures
    _traverse_json(request_data, request_root_id)
    _traverse_json(response_data, response_root_id)
    
    # 3. Add Cross-Record Edges (Request <-> Response link)
    edges_list.append((request_root_id, response_root_id))
    edges_list.append((response_root_id, request_root_id)) 
    
    # 4. Prepare PyTorch Geometric tensors
    num_nodes = node_id_counter
    
    if num_nodes == 0:
        return Data(x=torch.empty(0, 66), edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.tensor([label]))

    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    
    # 5. Extract features using the node map built during traversal
    x = extract_node_features(node_details_map)
    
    # 6. Final Graph Data object
    graph_data = Data(
        x=x, 
        edge_index=edge_index, 
        y=torch.tensor([label], dtype=torch.long)
    )
    
    return graph_data

if __name__ == '__main__':
    test_request = {
        "method": "POST",
        "endpoint": "/api/users",
        "payload": {
            "username": "admin",
            "password": "OR 1=1 --"
        }
    }
    
    test_response = {
        "status_code": 200,
        "body": {"success": True, "message": "Login successful"}
    }
    
    try:
        graph = build_graph_from_log(test_request, test_response, label=1)
        print("\n--- Deterministic Graph Summary ---")
        print(f"Total Nodes: {graph.num_nodes}")
        print(f"Total Edges: {graph.num_edges}")
        print(f"Graph Data:\n{graph}")
    except Exception as e:
        print(f"Error during construction: {e}")