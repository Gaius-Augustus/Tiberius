###############################################################################
# https://dzlab.github.io/dltips/en/tensorflow/callback-gpu-memory-consumption/
# adapted
###############################################################################

import tensorflow as tf
import json

class GPUMemoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, step_size, file_path, print_stats=True, **kwargs):
        """
        target_batches: A list of batch indices at which to record memory usage.
        print_stats: A boolean flag indicating whether to print memory usage statistics.
        """
        super().__init__(**kwargs)
        self.step_size = step_size
        self.print_stats = print_stats
        self.file_path = file_path

        self.gpus = tf.config.list_physical_devices('GPU')
        self.num_gpus = len(self.gpus)
        self.gpu_names = [f"GPU:{i}" for i in range(self.num_gpus)]

        self.memory_usage_peak = {gpu: [] for gpu in self.gpu_names}
        self.labels = []

    def _compute_memory_usage(self):
        for gpu_name in self.gpu_names:
            try:
                mem_info = tf.config.experimental.get_memory_info(gpu_name)
                # Convert bytes to GB and store in list.
                peak = round(mem_info["peak"] / (2**30), 3)
            except:
                peak = None
            self.memory_usage_peak[gpu_name].append(peak)

    def on_epoch_begin(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f"epoch {epoch} start")

    def on_train_batch_begin(self, batch, logs=None):
        #if batch in self.target_batches:
        if batch % self.step_size == 0:
            self._compute_memory_usage()
            self.labels.append(f"batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f"epoch {epoch} end")


    def on_train_end(self, logs=None):
        if self.file_path:
            data = {
                "gpu_usage_peak": self.memory_usage_peak,
                "labels": self.labels
            }
            with open(self.file_path, "w") as f:
                json.dump(data, f)