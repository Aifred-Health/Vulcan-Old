import tensorflow as tf

from datetime import datetime


def init_placeholders(feature_size, num_labels, batch_size = None):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    X_placeholder = tf.placeholder(tf.float32, shape=([batch_size,
                                                            feature_size]),
                                                            name = 'input')
    Y_placeholder = tf.placeholder(tf.float32, shape=([batch_size,
                                                            num_labels]),
                                                            name = 'truth')

    return X_placeholder, Y_placeholder


def get_timestamp():
    """Return a 14 digit timestamp."""
    return datetime.now().strftime('%Y%m%d%H%M%S_')
