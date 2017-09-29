# coding: utf-8
"""Train the original network using a feed dictionary."""

import tensorflow as tf
import original_cnn as oc

def placehoder_inputs(batch_size):
    """Generate palcehold variables to represent the input tensors.
    
    Args:
        batch_size: The batch size will be baked into both placeholders.
    
    Returns:
        data_placeholder: Data placeholder
        labels_placeholder: Labels placeholder
    """
    data_placeholder = tf.placeholder(tf.float32, shape = (batch_size, 1 * oc.BANDS_SIZE))
    label_placeholder = tf.placeholder(tf.int32, shape = (batch_size, 1 * oc.NUM_CLASSES))