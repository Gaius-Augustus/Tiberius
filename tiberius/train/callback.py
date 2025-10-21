import tensorflow as tf
from hidten.config import ModelConfig, with_config


class PrintLr(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        # logs['lr'] will be set by LearningRateScheduler
        lr = logs.get('lr')
        if lr is not None:
            print(f"Epoch {epoch+1}: Learning rate is {lr:.6f}")
        else:
            # fallback if you didn’t use a scheduler callback
            lr_t = self.model.optimizer.learning_rate
            # if it’s a schedule, call it on current iteration
            if isinstance(
                lr_t,
                tf.keras.optimizers.schedules.LearningRateSchedule,
            ):
                lr_t = lr_t(self.model.optimizer.iterations)
            print(
                f"\nEpoch {epoch+1}: "
                f"Learning rate is {tf.keras.backend.get_value(lr_t):.6f}"
            )


class BatchLearningRateScheduler(
    tf.keras.optimizers.schedules.LearningRateSchedule
):

    def __init__(
        self,
        peak: float = 0.1,
        warmup: int = 0,
        min_lr: float = 0.0001,
    ) -> None:
        super().__init__()
        self.peak = peak
        self.warmup = warmup
        self.total_batches = 0
        self.min_lr = min_lr

    def call(self, batch, logs=None) -> float:
        self.total_batches += 1
        if self.total_batches <= self.warmup:
            new_lr = self.total_batches * self.peak / self.warmup
        elif self.total_batches > self.warmup:
            new_lr = (self.total_batches-self.warmup)**(-1/2) * self.peak
        if new_lr > self.min_lr:
            return new_lr
        return self.min_lr


class WarmupExponentialDecayConfig(ModelConfig):
    peak_lr: float
    warmup_epochs: int
    decay_rate: float
    min_lr: float
    steps_per_epoch: int


@with_config(WarmupExponentialDecayConfig)
class WarmupExponentialDecay(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = WarmupExponentialDecayConfig(**kwargs)

    def __call__(self, step):
        epoch_int = tf.cast(
            step,
            dtype=tf.float32,
        ) // self.config.steps_per_epoch
        epoch = tf.cast(epoch_int, tf.float32)
        lr = tf.cond(
            epoch < self.config.warmup_epochs,
            lambda: self.config.peak_lr * (
                (epoch + 1) / tf.cast(self.config.warmup_epochs, tf.float32)
            ),
            lambda: tf.maximum(
                self.config.peak_lr * tf.pow(
                    self.config.decay_rate,
                    epoch - self.config.warmup_epochs + 1,
                ),
                self.config.min_lr,
            )
        )
        return lr


class ValidationCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_gen, save_path):
        super(ValidationCallback, self).__init__()
        self.best_val_loss = float('inf')
        self.save_path = save_path
        self.val_gen = val_gen

    def on_train_batch_end(self, batch, logs=None) -> None:
        if (batch + 1) % 3000 == 0:
            loss, acc = self.model.evaluate(self.val_gen, verbose=1)
            print(loss, acc)
            with open(self.save_path + '_log.txt', 'a+') as f_h:
                f_h.write(f'{loss}, {acc}\n')
            if loss < self.best_val_loss:
                print("Saving the model as the validation loss improved.")
                self.best_val_loss = loss
                self.model.save(self.save_path, save_traces=False)
                #self.model.save_weights(self.save_path)


class BatchSave(tf.keras.callbacks.Callback):

    def __init__(self, save_path: str, batch_number: int) -> None:
        super(BatchSave, self).__init__()
        self.save_path = save_path
        self.batch_number = batch_number
        self.prev_batch_numb = 0
        self.tf_old = tf.__version__ < "2.12.0"

    def on_train_batch_end(self, batch, logs=None) -> None:
        if (batch + 1) % self.batch_number == 0:
            self.prev_batch_numb += batch
            if self.tf_old:
                self.model.save(
                    self.save_path.format(self.prev_batch_numb),
                    save_traces=False,
                )
            else:
                self.model.save(
                    self.save_path.format(self.prev_batch_numb) + ".keras",
                )
