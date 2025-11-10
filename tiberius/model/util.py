import tensorflow as tf


def extract_nucleotides(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(
        x[0][..., :5] if isinstance(x, list) else x[..., :5],  # type: ignore
        tf.float32,
    )  # type: ignore
