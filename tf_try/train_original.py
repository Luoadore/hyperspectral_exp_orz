# coding: utf-8
"""Train the original network using a feed dictionary."""

import tensorflow as tf

def placehoder_inputs(batch_size):
    """Generate palcehold variables to represent the input tensors.
    
    Args:
        batch_size: The batch size will be baked into both placeholders.
    
    Returns:
        data_placeholder: Data placeholder
        labels_placeholder: Labels placeholder
    """
    