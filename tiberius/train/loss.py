import tensorflow as tf


class CCE_F1_Loss(tf.keras.Loss):

    def __init__(
        self,
        f1_factor: float,
        batch_size: int,
        output_dim: int = 15,
        include_reading_frame: bool = True,
        use_cee: bool = True,
    ) -> None:
        super().__init__(name="cee_f1", reduction=None)
        self.f1_factor = f1_factor
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.include_reading_frame = include_reading_frame
        self.use_cee = use_cee

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        B, T, D = tf.unstack(tf.shape(y_pred))
        y_true = tf.reshape(y_true, (B, T, 2, D // 2))
        y_pred = tf.reshape(y_pred, (B, T, 2, D // 2))
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.use_cee:
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                y_true, y_pred,
                from_logits=False,
            ))
        else:
            loss = 0

        if self.f1_factor > 0:
            dim = -3 if self.output_dim == 5 else 4
            if self.include_reading_frame:
                cds_pred = y_pred[:, :, :, dim:]
                cds_true = y_true[:, :, :, dim:]
            else:
                cds_pred = tf.reduce_sum(
                    y_pred[:, :, :, dim:],
                    axis=-1,
                )
                cds_true = tf.reduce_sum(
                    y_true[:, :, :, dim:],
                    axis=-1,
                )

            # Compute precision and recall for the specified class
            true_positives = tf.reduce_sum(cds_pred * cds_true, axis=1)
            predicted_positives = tf.reduce_sum(cds_pred, axis=1)
            possible_positives = tf.reduce_sum(cds_true, axis=1)
            any_positives = tf.cast(
                possible_positives > 0,
                possible_positives.dtype,
            )

            eps = tf.keras.backend.epsilon()
            precision = true_positives / (predicted_positives + eps)
            recall = true_positives  / (possible_positives + eps)

            # For the examples with positive class, maximize the F1 score
            f1_score = 2 * (precision * recall) / (precision + recall + eps)
            f1_loss = tf.reduce_mean((1 - f1_score) * any_positives)

            # For examples with no positive class, minimize false positive rate
            T = tf.cast(tf.shape(cds_pred)[1], cds_pred.dtype)
            fpr = tf.reduce_mean(
                cds_pred * tf.expand_dims(1 - any_positives, axis=1),
            )

            # Combine CCE loss and F1 score
            loss = loss + self.f1_factor * (f1_loss + fpr)

        return loss
