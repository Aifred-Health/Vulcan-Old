import tensorflow as tf


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
    images_placeholder = tf.placeholder(tf.float32, shape=([batch_size,
                                                            feature_size]),
                                                            name = 'input')
    labels_placeholder = tf.placeholder(tf.float32, shape=([batch_size,
                                                            num_labels]),
                                                            name = 'truth')

    return images_placeholder, labels_placeholder
