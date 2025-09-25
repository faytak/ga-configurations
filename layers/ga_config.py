"""
Unified configuration for GA (Geometric Algebra) layer types.

This module provides shared configuration constants for all GA layer types,
eliminating duplication across different layer implementations.
"""

# Base configuration that is common across all GA layer types
GA_GROUP_CONFIGS = {
    'P': {
        'dims_attr': ['dim_even', 'dim_odd'],
        'perm_attr': 'parity_weights_permutation'
    },
    'Q': {
        'dims_attr': ['dim_0', 'dim_1', 'dim_2', 'dim_3'],
        'perm_attr': 'weights_permutation'
    },
    'A': {
        'dims_attr': ['dim_01', 'dim_23'],
        'perm_attr': 'qt01_23_weights_permutation'
    },
    'B': {
        'dims_attr': ['dim_03', 'dim_12'],
        'perm_attr': 'qt03_12_weights_permutation'
    },
    'Triple': {
        'dims_attr': ['dim_triple0', 'dim_triple1', 'dim_triple2'],
        'perm_attr': 'triples_weights_permutation'
    }
}

# Layer-specific configurations that extend the base config
GA_LAYER_CONFIGS = {
    'linear': {
        'P': {'weight_init_dim': 2},
        'Q': {'weight_init_dim': 4},
        'A': {'weight_init_dim': 2},
        'B': {'weight_init_dim': 2},
        'Triple': {'weight_init_dim': 3}
    },
    'normalization': {
        'P': {'norm_fn_attr': 'norms_parity'},
        'Q': {'norm_fn_attr': 'norms_qt'},
        'A': {'norm_fn_attr': 'norms_qt01_23'},
        'B': {'norm_fn_attr': 'norms_qt03_12'},
        'Triple': {'norm_fn_attr': 'norms_triple'}
    },
    'silu': {
        'P': {'num_subspaces': 3, 'norm_method': 'parity_qs'},
        'Q': {'num_subspaces': 5, 'norm_method': 'qt_qs'},
        'A': {'num_subspaces': 3, 'norm_method': 'qt01_23_qs'},
        'B': {'num_subspaces': 3, 'norm_method': 'qt03_12_qs'},
        'Triple': {'num_subspaces': 4, 'norm_method': 'triple_qs'}
    },
    'geometric_product': {
        'P': {
            'product_paths_attr': 'parity_geometric_product_paths',
            'normalization_cls': 'PNormalization',
            'linear_cls': 'PLinear'
        },
        'Q': {
            'product_paths_attr': 'qt_geometric_product_paths',
            'normalization_cls': 'QNormalization',
            'linear_cls': 'QLinear'
        },
        'A': {
            'product_paths_attr': 'qt01_23_geometric_product_paths',
            'normalization_cls': 'ANormalization',
            'linear_cls': 'ALinear'
        },
        'B': {
            'product_paths_attr': 'qt01_23_geometric_product_paths',
            'normalization_cls': 'BNormalization',
            'linear_cls': 'BLinear'
        },
        'Triple': {
            'product_paths_attr': 'triples_geometric_product_paths',
            'normalization_cls': 'TripleNormalization',
            'linear_cls': 'TripleLinear'
        }
    }
}

# SiLU-specific dims_tensor configurations (special case due to scalar exclusion)
SILU_DIMS_TENSOR_CONFIGS = {
    'P': lambda algebra: [1, algebra.dim_even - 1, algebra.dim_odd],
    'Q': lambda algebra: [1, algebra.dim_0 - 1, algebra.dim_1, algebra.dim_2, algebra.dim_3],
    'A': lambda algebra: [1, algebra.dim_01 - 1, algebra.dim_23],
    'B': lambda algebra: [1, algebra.dim_03 - 1, algebra.dim_12],
    'Triple': lambda algebra: [1, algebra.dim_triple0 - 1, algebra.dim_triple1, algebra.dim_triple2]
}


def get_ga_config(group_type, layer_type):
    """
    Get configuration for a specific group type and layer type.
    
    Args:
        group_type: One of 'P', 'Q', 'A', 'B', 'Triple'
        layer_type: One of 'linear', 'normalization', 'silu', 'geometric_product'
    
    Returns:
        dict: Combined configuration for the specified group and layer types
    """
    print(f"Getting GA config for {group_type} and {layer_type}")
    
    if group_type not in GA_GROUP_CONFIGS:
        raise ValueError(f"group_type must be one of {list(GA_GROUP_CONFIGS.keys())}, got {group_type}")
    
    if layer_type not in GA_LAYER_CONFIGS:
        raise ValueError(f"layer_type must be one of {list(GA_LAYER_CONFIGS.keys())}, got {layer_type}")
    
    # Start with base configuration
    config = GA_GROUP_CONFIGS[group_type].copy()
    
    # Add layer-specific configuration
    config.update(GA_LAYER_CONFIGS[layer_type][group_type])
    
    return config


def get_silu_dims_tensor(algebra, group_type):
    """
    Get dims_tensor for SiLU layer based on group type.
    
    Args:
        algebra: Clifford algebra object
        group_type: One of 'P', 'Q', 'A', 'B', 'Triple'
    
    Returns:
        torch.Tensor: dims_tensor for the specified group type
    """
    import torch
    
    if group_type not in SILU_DIMS_TENSOR_CONFIGS:
        raise ValueError(f"group_type must be one of {list(SILU_DIMS_TENSOR_CONFIGS.keys())}, got {group_type}")
    
    return torch.tensor(SILU_DIMS_TENSOR_CONFIGS[group_type](algebra))


# Convenience functions for each layer type
def get_linear_config(group_type):
    """Get configuration for linear layer."""
    return get_ga_config(group_type, 'linear')


def get_normalization_config(group_type):
    """Get configuration for normalization layer."""
    return get_ga_config(group_type, 'normalization')


def get_silu_config(group_type):
    """Get configuration for SiLU layer."""
    return get_ga_config(group_type, 'silu')


def get_geometric_product_config(group_type):
    """Get configuration for geometric product layer."""
    return get_ga_config(group_type, 'geometric_product')
